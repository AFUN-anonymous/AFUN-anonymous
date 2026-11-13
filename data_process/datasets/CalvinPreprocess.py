"""
CALVIN preprocessing (static camera only).

Run on the 640x640 re-render made by tools/calvin_rerender_640.py; the image
size is read from the first NPZ file.

Raw data layout::

    {root}/{split}/
        episode_{frame_id:07d}.npz           # one timestep: rgb_static, depth_static (m), robot_obs, ...
        ep_start_end_ids.npy                 # (N, 2) episode boundary frame IDs
        lang_annotations/auto_lang_ann.npy   # language annotations + frame ranges

Used: rgb_static/depth_static, robot_obs[:3] (TCP position, world, m),
robot_obs[3:6] (TCP euler XYZ, for the flange), robot_obs[14] (gripper action,
-1 = close), language annotations, and the static camera pose from the CALVIN
env config. Gripper/tactile cameras and scene_obs (simulator state) are not used.

Intervals are anchored on language annotations and kept only when a
gripper-closed segment overlaps them (see detect_intervals). Camera extrinsics
are converted from PyBullet's OpenGL look-at convention to the CV convention.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, CameraInfo, EpisodeData


# ================================================================== #
# Calvin-specific helpers
# ================================================================== #

# Static camera parameters from the CALVIN env config (.hydra/merged_config.yaml),
# identical across environments A-D. width/height are only a fallback.
STATIC_CAM_CONFIG = {
    'fov': 10,        # vertical FOV in degrees
    'width': 200,
    'height': 200,
    'look_at': [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459],
    'look_from': [2.871459009488717, -2.166602199425597, 2.555159848480571],
    'up_vector': [0.4041403970338857, 0.22629790978217404, 0.8862616969685161],
}

# Control frequency from Calvin env config
CALVIN_FPS = 30.0


def _detect_gripper_closed_regions(gripper_action: np.ndarray) -> List[Tuple[int, int]]:
    """Detect contiguous frame ranges where the gripper is being held closed.

    CALVIN's gripper action signal (``robot_obs[14]``) is in {-1, +1}:
    -1 commands close, +1 commands open. So ``gripper_action < 0`` marks
    frames where the gripper is commanded closed.

    Returns list of (start, end) tuples in the same index space as input.
    """
    closed = gripper_action < 0
    regions: List[Tuple[int, int]] = []
    start = None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(gripper_action) - 1))
    return regions


def _compute_intrinsics_from_fov(fov_deg: float, width: int, height: int) -> Dict[str, float]:
    """Compute pinhole intrinsics from PyBullet vertical FOV."""
    fy = height / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    fx = fy  # aspect ratio = 1 (square image)
    return {'fx': fx, 'fy': fy, 'cx': width / 2.0, 'cy': height / 2.0}


def _compute_T_world_to_cam(look_from, look_at, up_vector) -> np.ndarray:
    """Compute 4x4 world-to-camera transform from PyBullet camera config.

    PyBullet uses OpenGL convention (camera looks along -Z, Y up).
    The pipeline expects CV convention (camera looks along +Z, Y down).
    """
    look_from = np.asarray(look_from, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    up = np.asarray(up_vector, dtype=np.float64)

    # OpenGL view matrix construction
    forward = look_at - look_from
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_new = np.cross(right, forward)

    # OpenGL convention: rows are [right, up, -forward]
    R_gl = np.array([right, up_new, -forward])
    t_gl = -R_gl @ look_from

    # Convert OpenGL -> CV convention: flip Y and Z axes
    gl_to_cv = np.diag([1.0, -1.0, -1.0])
    R_cv = gl_to_cv @ R_gl
    t_cv = gl_to_cv @ t_gl

    T = np.eye(4)
    T[:3, :3] = R_cv
    T[:3, 3] = t_cv
    return T


# ================================================================== #
# CalvinPreprocess
# ================================================================== #

class CalvinPreprocess(RoboDatasetBase):
    """CALVIN dataset preprocessor.

    Uses only the static camera (gripper camera is arm-mounted and moves
    during manipulation, violating the static-camera assumption).

    Intervals use both signals for quality: language annotations provide
    task descriptions and approximate boundaries, gripper action signal
    validates that real manipulation occurred and expands boundaries to
    capture the full motion. Annotations with no gripper activity are
    rejected. Near-duplicate annotations are merged.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.root = Path(dataset.root)
        self.splits = list(dataset.get('splits', ['training', 'validation']))
        self.gripper_offset_m = dataset.get('gripper_offset_m', 0.10)
        self._extract_depth = dataset.get('extract_depth', True)
        self.min_interval_time = dataset.get('min_interval_time', 0.5)

        # Task filter -- if a filter YAML is specified, only keep listed tasks
        task_filter_path = dataset.get('task_filter', None)
        if task_filter_path is not None:
            with open(task_filter_path) as f:
                self._keep_tasks = set(yaml.safe_load(f)['keep_tasks'])
            print(f"[Calvin] Task filter: keeping {len(self._keep_tasks)} tasks")
        else:
            self._keep_tasks = None

        # Detect image resolution from first NPZ (supports re-rendered 640x640)
        sample_split = self.splits[0]
        sample_dir = self.root / sample_split
        sample_npz = next(sample_dir.glob('episode_*.npz'), None)
        if sample_npz is not None:
            with np.load(str(sample_npz)) as data:
                h, w = data['rgb_static'].shape[:2]
        else:
            h, w = STATIC_CAM_CONFIG['height'], STATIC_CAM_CONFIG['width']

        # Precompute camera parameters (shared across all episodes)
        self._intrinsics = _compute_intrinsics_from_fov(
            STATIC_CAM_CONFIG['fov'],
            w,
            h,
        )
        self._T_world_to_cam = _compute_T_world_to_cam(
            STATIC_CAM_CONFIG['look_from'],
            STATIC_CAM_CONFIG['look_at'],
            STATIC_CAM_CONFIG['up_vector'],
        )

        # Preload language annotations per split
        self._lang_annotations: Dict[str, Dict] = {}
        for split in self.splits:
            ann_path = self.root / split / 'lang_annotations' / 'auto_lang_ann.npy'
            if ann_path.exists():
                self._lang_annotations[split] = np.load(
                    str(ann_path), allow_pickle=True
                ).item()

        print(f"[Calvin] root={self.root}  splits={self.splits}")
        print(f"[Calvin] intrinsics: fx={self._intrinsics['fx']:.1f} "
              f"fy={self._intrinsics['fy']:.1f}")

    # ------------------------------------------------------------------ #
    # Abstract implementations
    # ------------------------------------------------------------------ #

    def discover_episodes(self) -> List[str]:
        """Discover episodes from ep_start_end_ids.npy in each split.

        Returns episode paths as "{split}/{abs_start}_{abs_end}".
        """
        episodes = []
        for split in self.splits:
            split_dir = self.root / split
            ids_path = split_dir / 'ep_start_end_ids.npy'
            if not ids_path.exists():
                print(f"[Calvin] WARNING: {ids_path} not found, skipping split")
                continue
            ep_ids = np.load(str(ids_path))
            for start, end in ep_ids:
                episodes.append(f"{split}/{int(start)}_{int(end)}")
        return sorted(episodes)

    def get_episode_id(self, episode_path: str) -> str:
        """Return 'calvin_{split}_{start}_{end}' as a unique identifier."""
        split, frame_range = episode_path.split('/')
        return f"calvin_{split}_{frame_range}"

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load robot_obs for all frames; images loaded on demand.

        Frame indexing is RELATIVE (0 to num_frames-1). The absolute
        frame offset is stored in extra['abs_start'] for NPZ file lookup.
        """
        split, frame_range = episode_path.split('/')
        abs_start, abs_end = map(int, frame_range.split('_'))
        split_dir = self.root / split
        num_frames = abs_end - abs_start + 1

        # Preload robot_obs for all frames (small: 15 floats per frame)
        robot_obs = np.zeros((num_frames, 15), dtype=np.float64)
        for i in tqdm(range(num_frames), desc=f"  Loading {episode_path}", leave=False):
            abs_idx = abs_start + i
            npz_path = split_dir / f"episode_{abs_idx:07d}.npz"
            data = np.load(str(npz_path))
            robot_obs[i] = data['robot_obs']
            data.close()

        # Static camera info (gripper camera excluded -- it is arm-mounted)
        cam_info = CameraInfo(
            cam_id='static',
            intrinsics=self._intrinsics,
            T_base_to_cam=self._T_world_to_cam,
            extrinsic_source='calvin_env_config',
            intrinsic_source='calvin_env_fov',
        )

        return EpisodeData(
            num_frames=num_frames,
            fps=CALVIN_FPS,
            camera_infos=[cam_info],
            extra={
                'split': split,
                'split_dir': str(split_dir),
                'abs_start': abs_start,
                'abs_end': abs_end,
                'robot_obs': robot_obs,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Language-anchored, gripper-validated interval detection.

        Strategy:
          1. Detect gripper-closed regions from robot_obs gripper action signal.
          2. For each language annotation, find gripper-closed regions that
             overlap with it (with a small margin for alignment slack).
          3. Reject annotations with NO gripper activity -- not real manipulations.
          4. Expand the interval to the union of the annotation window and all
             overlapping gripper-closed regions (captures the full motion).
          5. Merge near-duplicate annotations that map to the same frame range.

        Camera filtering: only the static camera is included (see
        load_episode). The gripper camera is arm-mounted and excluded.
        All Calvin tasks are rigid-body (blocks, switches, drawers,
        sliders) -- no deformable-object filtering needed.
        """
        split = episode_data.extra['split']
        abs_start = episode_data.extra['abs_start']
        abs_end = episode_data.extra['abs_end']
        num_frames = episode_data.num_frames
        fps = episode_data.fps
        min_frames = max(1, int(self.min_interval_time * fps))

        # --- Step 1: gripper-closed regions from the action signal ---
        gripper_action = episode_data.extra['robot_obs'][:, 14]
        gripper_regions = _detect_gripper_closed_regions(gripper_action)

        # --- Step 2-3: match language annotations to gripper activity ---
        ann = self._lang_annotations.get(split)
        if ann is None:
            print(f"  [Calvin] No language annotations for split '{split}'")
            return []

        texts = ann['language']['ann']
        tasks = ann['language'].get('task', [None] * len(texts))
        frame_ranges = ann['info']['indx']
        overlap_margin = int(0.5 * fps)  # 0.5s margin for alignment slack

        # Collect validated, expanded intervals
        raw: List[Tuple[int, int, str]] = []  # (expanded_start, expanded_end, text)
        for text, task, (ann_start, ann_end) in zip(texts, tasks, frame_ranges):
            ann_start, ann_end = int(ann_start), int(ann_end)

            # Task filter -- skip annotations for excluded tasks
            if self._keep_tasks is not None and task is not None:
                if task not in self._keep_tasks:
                    continue

            # Skip annotations outside this episode
            if ann_end < abs_start or ann_start > abs_end:
                continue

            # Convert to relative indices, clamp to episode
            rel_start = max(ann_start, abs_start) - abs_start
            rel_end = min(ann_end, abs_end) - abs_start

            # Find gripper-closed regions overlapping with [rel_start - margin, rel_end + margin]
            search_start = max(0, rel_start - overlap_margin)
            search_end = min(num_frames - 1, rel_end + overlap_margin)
            overlapping = [
                (gs, ge) for gs, ge in gripper_regions
                if gs <= search_end and ge >= search_start
            ]

            if not overlapping:
                # No gripper activity near this annotation -- skip
                print(f"    [Calvin] Skipping '{text}' [{rel_start}-{rel_end}]: "
                      f"no gripper activity")
                continue

            # Expand interval to union of annotation + overlapping gripper regions
            expanded_start = min(rel_start, min(gs for gs, ge in overlapping))
            expanded_end = max(rel_end, max(ge for gs, ge in overlapping))
            expanded_start = max(0, expanded_start)
            expanded_end = min(num_frames - 1, expanded_end)

            if expanded_end - expanded_start + 1 < min_frames:
                continue

            raw.append((expanded_start, expanded_end, text))

        # --- Step 5: merge overlapping intervals, keep longest language ---
        raw.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int, str]] = []
        for start, end, text in raw:
            if merged and start <= merged[-1][1]:
                # Overlapping with previous -- extend end, keep longer description
                prev_s, prev_e, prev_t = merged[-1]
                new_end = max(prev_e, end)
                new_text = text if len(text) > len(prev_t) else prev_t
                merged[-1] = (prev_s, new_end, new_text)
            else:
                merged.append((start, end, text))

        # --- Assign obs_frame (pre-action) and contact_frame (action start) ---
        # 2 s lookback so obs_frame captures the scene before manipulation begins.
        obs_lead = int(2.0 * fps)
        intervals: List[Interval] = []
        for i, (start, end, text) in enumerate(merged):
            if i == 0:
                obs = 0  # first interval: episode start
            else:
                obs = max(0, start - obs_lead)
                prev_end = merged[i - 1][1]
                if prev_end < start:
                    obs = max(obs, prev_end + 1)

            intervals.append(Interval(
                start=start,
                end=end,
                language=text,
                obs_frame=obs,
                contact_frame=start,
            ))

        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load RGB and depth from NPZ for the static camera.

        frame_idx is relative (0-based). Depth is converted from meters
        to millimeters.
        """
        abs_idx = episode_data.extra['abs_start'] + frame_idx
        split_dir = Path(episode_data.extra['split_dir'])
        npz_path = split_dir / f"episode_{abs_idx:07d}.npz"

        data = np.load(str(npz_path))
        rgb = data['rgb_static'].copy()  # (H, W, 3) uint8

        depth = None
        if self._extract_depth:
            depth_m = data['depth_static']  # (H, W) float32, meters
            depth = (depth_m * 1000.0).astype(np.float32)  # -> millimeters

        data.close()
        return rgb, depth

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """TCP position from robot_obs[:3]. Already in world meters."""
        return episode_data.extra['robot_obs'][frame_idx, :3].copy()

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Flange = TCP position minus gripper offset along TCP z-axis."""
        robot_obs = episode_data.extra['robot_obs'][frame_idx]
        tcp_pos = robot_obs[:3]
        tcp_euler = robot_obs[3:6]
        rot = R.from_euler('xyz', tcp_euler).as_matrix()
        return tcp_pos - self.gripper_offset_m * rot[:, 2]

    def close_episode(self, episode_data: EpisodeData) -> None:
        """No persistent resources to release (NPZ files are loaded on demand)."""
        episode_data.extra.clear()
