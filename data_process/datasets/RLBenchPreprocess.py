"""
RLBench preprocessing (PerAct 18-task data rendered at 512x512).

The data is made by tools/generate_rlbench_512.sh. Layout::

    {root}/data/{split}/{task}/all_variations/episodes/episodeN/
        {cam}_rgb/{t}.png, {cam}_depth/{t}.png   # depth: 24-bit RGB-encoded
        low_dim_obs.pkl                          # per-step Observation objects
        variation_descriptions.pkl               # language descriptions

Only the static cameras (left_shoulder, right_shoulder, overhead, front) are
used; the wrist camera moves with the arm. Intervals are the spans where
gripper_open < 0.5, with obs_frame 1 s earlier; the whole episode is used when
the gripper never closes. Intrinsics come from misc['{cam}_camera_intrinsics']
(abs() of the negative CoppeliaSim focal lengths) and extrinsics are the
inverse of misc['{cam}_camera_extrinsics'] (camera pose in world). Depth is
decoded with the near/far planes and converted to mm. Simulator masks and
object states are not used.
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as Rot

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, CameraInfo, EpisodeData


logger = logging.getLogger(__name__)


# ================================================================== #
# RLBench pickle stubs (avoid requiring full rlbench package)
# ================================================================== #

class _Observation:
    """Stub for rlbench.backend.observation.Observation."""
    def __setstate__(self, state):
        self.__dict__.update(state)

class _Demo(list):
    """Stub for rlbench.demo.Demo (stores observations in _observations attr)."""
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

class _RLBenchUnpickler(pickle.Unpickler):
    """Custom unpickler that resolves RLBench classes without the full package."""
    def find_class(self, module, name):
        if module == 'rlbench.demo' and name == 'Demo':
            return _Demo
        if module == 'rlbench.backend.observation' and name == 'Observation':
            return _Observation
        return super().find_class(module, name)

def _load_low_dim_obs(pkl_path: str) -> list:
    """Load low_dim_obs.pkl and return the list of Observation objects."""
    with open(pkl_path, 'rb') as f:
        demo = _RLBenchUnpickler(f).load()
    # Demo stores observations in _observations attribute
    return demo._observations


# ================================================================== #
# RLBench-specific constants
# ================================================================== #

# CoppeliaSim runs at 20 Hz by default for RLBench
RLBENCH_FPS = 20.0

# Panda TCP offset from flange
PANDA_TCP_OFFSET_M = 0.1034

# Static cameras (wrist is arm-mounted and moves)
STATIC_CAMERAS = ['left_shoulder', 'right_shoulder', 'overhead', 'front']

# Camera name -> misc dict key mapping
# RLBench stores camera params with "_camera" suffix in the misc dict
CAM_MISC_SUFFIX = '_camera'


# ================================================================== #
# Helpers
# ================================================================== #

def _decode_depth_png(depth_rgb: np.ndarray, near: float, far: float) -> np.ndarray:
    """Decode 24-bit RGB-encoded depth PNG to meters.

    RLBench encodes depth into 3 channels: depth_buf = (R*65536 + G*256 + B) / (2^24 - 1).
    Real depth = near + depth_buf * (far - near).

    Returns float32 array in meters.
    """
    depth_rgb = depth_rgb.astype(np.float64)
    depth_buf = (depth_rgb[:, :, 0] * 65536 + depth_rgb[:, :, 1] * 256 + depth_rgb[:, :, 2]) / (2**24 - 1)
    depth_m = near + depth_buf * (far - near)
    return depth_m.astype(np.float32)


def _detect_closed_intervals(
    gripper_open: np.ndarray,
    min_len: int,
) -> List[Tuple[int, int]]:
    """Detect contiguous intervals where gripper is closed (gripper_open < 0.5).

    Returns list of (start, end) tuples (inclusive).
    """
    if gripper_open.size == 0:
        return []
    closed = gripper_open < 0.5
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, is_closed in enumerate(closed):
        if is_closed and start is None:
            start = idx
        elif not is_closed and start is not None:
            if idx - start >= min_len:
                intervals.append((start, idx - 1))
            start = None
    if start is not None and len(gripper_open) - start >= min_len:
        intervals.append((start, len(gripper_open) - 1))
    return intervals


# ================================================================== #
# RLBenchPreprocess
# ================================================================== #

class RLBenchPreprocess(RoboDatasetBase):
    """RLBench (PerAct pre-rendered) dataset preprocessor.

    Processes pre-rendered RGB+depth PNG images with camera parameters
    stored in per-timestep misc dicts. Uses only static cameras
    (left_shoulder, right_shoulder, overhead, front). Wrist camera is
    arm-mounted and excluded.

    All RLBench tasks are rigid-body manipulation.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.root = Path(dataset.root)
        self.cameras = list(dataset.get('cameras', STATIC_CAMERAS))
        self.gripper_offset_m = dataset.get('gripper_offset_m', PANDA_TCP_OFFSET_M)
        self.min_interval_time = dataset.get('min_interval_time', 0.3)
        self.split = dataset.get('split', 'train')

        print(f"[RLBench] root={self.root}  cameras={self.cameras}")

    # ------------------------------------------------------------------ #
    # Abstract implementations
    # ------------------------------------------------------------------ #

    def discover_episodes(self) -> List[str]:
        """Discover episodes by finding low_dim_obs.pkl files under every task."""
        episodes = []
        data_dir = self.root / 'data' / self.split
        task_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir()) if data_dir.exists() else []

        for task_dir in task_dirs:
            episodes_dir = task_dir / 'all_variations' / 'episodes'
            if not episodes_dir.exists():
                continue
            for ep_dir in sorted(episodes_dir.iterdir()):
                if (ep_dir / 'low_dim_obs.pkl').exists():
                    rel = ep_dir.relative_to(self.root)
                    episodes.append(str(rel))

        print(f"[RLBench] Discovered {len(episodes)} episodes")
        return episodes

    def get_episode_id(self, episode_path: str) -> str:
        """Return 'rlbench_{task}_{episode}' as unique identifier."""
        parts = Path(episode_path).parts
        # parts like: ('data', 'train', 'open_drawer', 'all_variations', 'episodes', 'episode30')
        task = parts[2] if len(parts) > 2 else 'unknown'
        ep_name = parts[-1] if len(parts) > 0 else 'unknown'
        return f"rlbench_{task}_{ep_name}"

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load low_dim_obs.pkl and build camera infos from misc dict."""
        ep_dir = self.root / episode_path

        # Load pickled observations (uses custom unpickler to avoid rlbench dependency)
        obs_list = _load_low_dim_obs(str(ep_dir / 'low_dim_obs.pkl'))

        num_frames = len(obs_list)

        # Load language descriptions
        lang_path = ep_dir / 'variation_descriptions.pkl'
        language = None
        if lang_path.exists():
            with open(lang_path, 'rb') as f:
                descs = pickle.load(f)
            if descs:
                language = descs[0]  # use first variation description

        # Extract gripper state and pose for all frames
        gripper_open = np.array([obs.gripper_open for obs in obs_list], dtype=np.float64)
        gripper_pose = np.array([obs.gripper_pose for obs in obs_list], dtype=np.float64)  # (N, 7)

        # Build camera infos from frame 0 misc dict (static cameras have constant params)
        misc = obs_list[0].misc
        camera_infos = []
        cam_depth_params = {}  # cam_id -> {near, far}

        for cam in self.cameras:
            cam_key = cam + CAM_MISC_SUFFIX
            intrinsics_key = f'{cam_key}_intrinsics'
            extrinsics_key = f'{cam_key}_extrinsics'
            near_key = f'{cam_key}_near'
            far_key = f'{cam_key}_far'

            if (intrinsics_key not in misc or extrinsics_key not in misc
                    or near_key not in misc or far_key not in misc):
                logger.warning(f"[RLBench] Missing camera params for {cam} in {episode_path}")
                continue

            K = misc[intrinsics_key]  # (3, 3) -- may have negative fx, fy
            T_cam_in_world = misc[extrinsics_key]  # (4, 4) camera pose in world

            # CoppeliaSim uses negative focal lengths -- take abs
            fx = abs(float(K[0, 0]))
            fy = abs(float(K[1, 1]))
            cx = float(K[0, 2])
            cy = float(K[1, 2])

            # Validate intrinsics
            if fx < 1 or fy < 1:
                logger.warning(f"[RLBench] Implausible intrinsics for {cam}: fx={fx}, fy={fy}")
                continue

            # Invert to get T_world_to_cam (= T_base_to_cam since base ~ world in RLBench)
            T_base_to_cam = np.linalg.inv(T_cam_in_world).astype(np.float64)
            # raw: T_cam_in_world (camera pose) -> pipeline: T_base_to_cam (inverted)

            # Validate rotation
            rot_det = np.linalg.det(T_base_to_cam[:3, :3])
            if abs(rot_det - 1.0) > 0.01:
                logger.warning(f"[RLBench] Camera {cam} rotation det={rot_det:.4f}")
                continue

            cam_info = CameraInfo(
                cam_id=cam,
                intrinsics={'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
                T_base_to_cam=T_base_to_cam,
                extrinsic_source='rlbench_misc_extrinsics_inverted',
                intrinsic_source='rlbench_misc_intrinsics_abs',
            )
            camera_infos.append(cam_info)

            # Store depth decoding params
            near = float(misc[near_key])
            far = float(misc[far_key])
            cam_depth_params[cam] = {'near': near, 'far': far}

        if not camera_infos:
            logger.warning(f"[RLBench] No valid cameras for {episode_path}")

        return EpisodeData(
            num_frames=num_frames,
            fps=RLBENCH_FPS,
            camera_infos=camera_infos,
            extra={
                'ep_dir': str(ep_dir),
                'gripper_open': gripper_open,
                'gripper_pose': gripper_pose,
                'language': language,
                'cam_depth_params': cam_depth_params,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Detect manipulation intervals from gripper open/close transitions.

        All RLBench tasks are rigid-body. Only static cameras are used
        (wrist camera excluded in load_episode).
        """
        gripper_open = episode_data.extra['gripper_open']
        language = episode_data.extra['language']
        num_frames = episode_data.num_frames
        min_frames = max(1, int(self.min_interval_time * episode_data.fps))

        closed_intervals = _detect_closed_intervals(gripper_open, min_frames)

        intervals: List[Interval] = []

        if closed_intervals:
            for start, end in closed_intervals:
                # obs_frame: before gripper closes (approach phase)
                obs_lead = int(1.0 * episode_data.fps)
                obs_frame = max(0, start - obs_lead)

                intervals.append(Interval(
                    start=start,
                    end=end,
                    language=language,
                    obs_frame=obs_frame,
                    contact_frame=start,
                ))
        else:
            # No gripper closing detected -- use full episode as one interval
            # (some tasks like push don't close gripper)
            intervals.append(Interval(
                start=0,
                end=num_frames - 1,
                language=language,
                obs_frame=0,
                contact_frame=min(int(0.3 * num_frames), num_frames - 1),
            ))

        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load RGB and decode depth from PNG files.

        Depth PNGs are 24-bit RGB encoded. Decoded using near/far planes
        from camera misc params, then converted to millimeters.
        """
        ep_dir = Path(episode_data.extra['ep_dir'])

        # RGB
        rgb_path = ep_dir / f'{cam_id}_rgb' / f'{frame_idx}.png'
        rgb = np.array(_load_png(rgb_path))  # (H, W, 3) uint8

        # Depth -- 24-bit RGB encoded PNG
        depth_path = ep_dir / f'{cam_id}_depth' / f'{frame_idx}.png'
        depth_rgb = np.array(_load_png(depth_path))  # (H, W, 3) uint8

        cam_params = episode_data.extra['cam_depth_params'][cam_id]
        depth_m = _decode_depth_png(depth_rgb, cam_params['near'], cam_params['far'])
        depth_mm = (depth_m * 1000.0).astype(np.float32)  # raw: meters -> pipeline: mm

        return rgb, depth_mm

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """TCP position from gripper_pose[:3]. Already in world meters."""
        return episode_data.extra['gripper_pose'][frame_idx, :3].copy()

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Flange = TCP position minus gripper offset along TCP z-axis."""
        pose = episode_data.extra['gripper_pose'][frame_idx]
        tcp_pos = pose[:3].copy()
        tcp_quat = pose[3:7]  # [qx, qy, qz, qw] -- scipy convention
        rot = Rot.from_quat(tcp_quat).as_matrix()
        flange = tcp_pos - self.gripper_offset_m * rot[:, 2]
        return flange  # raw: meters -> pipeline: meters (no conversion needed)

    def close_episode(self, episode_data: EpisodeData) -> None:
        """No persistent resources to release (images loaded on demand)."""
        episode_data.extra.clear()


def _load_png(path: Path) -> np.ndarray:
    """Load a PNG image as numpy array."""
    from PIL import Image
    return np.array(Image.open(str(path)))
