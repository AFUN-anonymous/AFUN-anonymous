"""
AgiBot World Beta preprocessing from pre-cut action clips.

The data is already cut into action clips by an external pipeline. Each clip
folder is one episode with one interval; only clips listed in ``matched_clips``
are processed. Depth is not read here: mainprocess computes DA3 video depth
(stage sam3_mask_traj_depth).

Data layout
-----------
Clips live at::

    {clip_root}/{clip_name}/head/
        rgb.mp4            - 640x480 RGB, 15 fps
        caption.pickle     - action_text, skill, task_name, ...

Camera parameters (each root in ``param_roots`` is checked in order)::

    {param_root}/{task_id}/{episode_id}/parameters/camera/

Clip naming convention::

    {task_id}_{episode_id}_{action_idx}_{start_half}_{end_half}
    e.g. 327_648642_a0_18_93

Frame mapping: clip frame i -> original frame = start_half * 2 + i * 2
(clips are 2x downsampled from original AgiBotWorld framerate).
"""

import json
import pickle
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import av

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, CameraInfo, EpisodeData, TrajectoryPoint, Motion


# ================================================================== #
# Helpers
# ================================================================== #

class _VideoReader:
    """Sequential PyAV video reader with persistent decode generator.

    ``container.decode(stream)`` yields frames from the container's
    current packet position. To support sequential reads without
    re-decoding from the start, we hold one generator (``_decoder``)
    open across reads and advance it to the requested frame index.
    A backward seek reopens the container and starts a fresh generator.
    """

    def __init__(self, video_path: Path, cache_size: int = 50):
        self.video_path = video_path
        self.container = None
        self.stream = None
        self._decoder = None
        self._current_idx: int = -1
        self._cache: Dict[int, np.ndarray] = {}
        self._cache_size = cache_size
        self._num_frames: Optional[int] = None

    def _open(self):
        if self.container is not None:
            self.container.close()
        self.container = av.open(str(self.video_path))
        self.stream = self.container.streams.video[0]
        self._decoder = self.container.decode(self.stream)
        self._current_idx = -1

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx in self._cache:
            return self._cache[frame_idx]

        if self.container is None or frame_idx <= self._current_idx:
            self._open()

        for frame in self._decoder:
            self._current_idx += 1
            img = frame.to_ndarray(format='rgb24')

            if len(self._cache) >= self._cache_size:
                oldest = min(self._cache)
                del self._cache[oldest]
            self._cache[self._current_idx] = img

            if self._current_idx == frame_idx:
                return img

        raise ValueError(
            f"Could not read frame {frame_idx} from {self.video_path}"
        )

    @property
    def num_frames(self) -> int:
        """Count total frames (cached after first call)."""
        if self._num_frames is not None:
            return self._num_frames
        if self.container is None:
            self._open()
        # Try container metadata first
        if self.stream.frames > 0:
            self._num_frames = self.stream.frames
            return self._num_frames
        # Fallback: decode and count
        count = 0
        self._open()
        for _ in self._decoder:
            count += 1
        self._open()  # reset for subsequent reads
        self._num_frames = count
        return self._num_frames

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None
        self._decoder = None
        self._cache.clear()


_CLIP_RE = re.compile(
    r'^(\d+)_(\d+)_(a\d+)_(\d+)_(\d+)$'
)


def _parse_clip_name(clip_name: str) -> Dict[str, Any]:
    """Parse clip folder name into components.

    Returns dict with task_id, episode_id, action_idx,
    start_half, end_half.
    """
    m = _CLIP_RE.match(clip_name)
    if not m:
        raise ValueError(f"Cannot parse clip name: {clip_name}")
    return {
        'task_id': m.group(1),
        'episode_id': m.group(2),
        'action_idx': m.group(3),
        'start_half': int(m.group(4)),
        'end_half': int(m.group(5)),
    }


def _make_T_world_to_cam(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """Build 4x4 T_world_to_cam from camera-to-world (R, t).

    AgiBot convention: P_world = R @ P_cam + t
    Inverse: P_cam = R^T @ (P_world - t)
    """
    T = np.eye(4)
    T[:3, :3] = R_c2w.T
    T[:3, 3] = -R_c2w.T @ t_c2w
    return T


# ================================================================== #
# AgiBotPreprocess
# ================================================================== #

class AgiBotPreprocess(RoboDatasetBase):
    """AgiBot (pre-cut clips) dataset preprocessor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.clip_root = Path(dataset.clip_root)
        self.param_roots = [Path(p) for p in dataset.param_roots]

        # Load matched clips list (action-level filtering)
        matched_clips_path = Path(dataset.matched_clips)
        with open(matched_clips_path) as f:
            self._matched_clips: List[str] = json.load(f)

    # ------------------------------------------------------------------ #
    # Episode listing
    # ------------------------------------------------------------------ #

    def discover_episodes(self) -> List[str]:
        """Each clip folder is an episode. Filter by matched_clips only."""
        paths: List[str] = []
        for clip_name in sorted(self._matched_clips):
            clip_path = self.clip_root / clip_name
            if clip_path.is_dir():
                paths.append(str(clip_path))
        return paths

    def get_episode_id(self, episode_path: str) -> str:
        """Episode ID = clip folder name (e.g. 327_648642_a0_18_93)."""
        return Path(episode_path).name

    # ------------------------------------------------------------------ #
    # Episode loading
    # ------------------------------------------------------------------ #

    def load_episode(self, episode_path: str) -> EpisodeData:
        clip_dir = Path(episode_path)
        clip_name = clip_dir.name
        parsed = _parse_clip_name(clip_name)

        task_id = parsed['task_id']
        episode_id = parsed['episode_id']
        start_half = parsed['start_half']

        # --- Video reader ---
        video_path = clip_dir / "head" / "rgb.mp4"
        reader = _VideoReader(video_path)
        num_frames = reader.num_frames

        # --- Caption / action text ---
        caption_path = clip_dir / "head" / "caption.pickle"
        with open(caption_path, 'rb') as f:
            caption_data = pickle.load(f)
        action_text = caption_data.get('action_text', '')

        # --- Camera parameters ---
        intrinsics, extrinsic_T, distortion = self._load_head_camera(
            task_id, episode_id, start_half * 2
        )

        cam = CameraInfo(
            cam_id="head_color",
            intrinsics=intrinsics,
            T_base_to_cam=extrinsic_T,
            extrinsic_source="aligned_per_frame",
            intrinsic_source="scaled_intrinsic_params",
            distortion=distortion,
        )

        return EpisodeData(
            num_frames=num_frames,
            fps=15.0,
            camera_infos=[cam],
            extra={
                'action_text': action_text,
                'video_reader': reader,
            },
        )

    # ------------------------------------------------------------------ #
    # Interval detection
    # ------------------------------------------------------------------ #

    # Minimum frames needed: compute_motion requires >= int(fps) * 0.5
    _MIN_FRAMES = 8

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Each clip IS one interval. No gripper detection needed."""
        if episode_data.num_frames < self._MIN_FRAMES:
            print(f"  Clip too short ({episode_data.num_frames} frames), skipping.")
            return []
        return [Interval(
            start=0,
            end=episode_data.num_frames - 1,
            language=episode_data.extra['action_text'],
            obs_frame=0,
            contact_frame=0,
        )]

    # ------------------------------------------------------------------ #
    # Override trajectory: use image center as seed point
    # ------------------------------------------------------------------ #

    def compute_trajectory(
        self, episode_data: EpisodeData, interval: Interval,
        cam_info: CameraInfo,
    ) -> List[TrajectoryPoint]:
        """Image-center placeholder, written only to satisfy the
        trajectory.json output format.

        AgiBot does not consume this value; the actual trajectory is
        obtained from tracking in mainprocess.
        """
        cx = int(cam_info.intrinsics['cx'])
        cy = int(cam_info.intrinsics['cy'])
        pts: List[TrajectoryPoint] = []
        for idx in range(interval.start, interval.end + 1):
            pts.append(TrajectoryPoint(
                frame_idx=idx,
                position_3d=[0.0, 0.0, 0.0],
                position_pixel=[cx, cy],
            ))
        return pts

    def compute_motion(
        self, trajectory: List[TrajectoryPoint], fps: float,
    ) -> Motion:
        """Zero-motion placeholder, written only to satisfy the
        trajectory.json output format.

        AgiBot does not consume this value; the real motion is
        obtained from tracking in mainprocess.
        """
        pixel = trajectory[0].position_pixel if trajectory else None
        return Motion(
            motion_3d=[0.0, 0.0, 0.0],
            motion_2d_start=pixel,
            motion_2d_end=pixel,
        )

    # ------------------------------------------------------------------ #
    # Frame / depth extraction
    # ------------------------------------------------------------------ #

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RGB from the clip video; no depth (computed later in mainprocess)."""
        return episode_data.extra['video_reader'].get_frame(frame_idx), None

    # ------------------------------------------------------------------ #
    # Gripper / flange (dummy -- not used for this dataset)
    # ------------------------------------------------------------------ #

    # Required by the base class but not consumed by AgiBot's pipeline.
    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        return np.zeros(3)

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        return np.zeros(3)

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #

    def close_episode(self, episode_data: EpisodeData) -> None:
        reader = episode_data.extra.get('video_reader')
        if reader is not None:
            reader.close()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _find_param_dir(self, task_id: str, episode_id: str) -> Path:
        """Find camera parameter directory, checking multiple roots."""
        for root in self.param_roots:
            d = root / task_id / episode_id / "parameters" / "camera"
            if d.is_dir():
                return d
        raise FileNotFoundError(
            f"Camera params not found for task={task_id} ep={episode_id} "
            f"in {[str(r) for r in self.param_roots]}"
        )

    def _load_head_camera(
        self, task_id: str, episode_id: str, obs_original_frame: int
    ) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Any]]:
        """Load head camera intrinsics, extrinsic at obs frame, and distortion.

        Returns (intrinsics_dict, T_world_to_cam_4x4, distortion_dict).
        """
        params_dir = self._find_param_dir(task_id, episode_id)

        # --- Intrinsics ---
        with open(params_dir / "head_intrinsic_params.json") as f:
            intr = json.load(f)['intrinsic']

        # Rescale intrinsics: AgiBot records at higher sensor resolution
        # This version of AgiBotWorld-Beta has intrinsic scalign issue! The authors did not update the principal point when downsampling. 
        # We fix it here by assuming the original principal point is at the center of the original sensor, and rescaling it according to the downsampled image size.
        ppx = intr.get('ppx', 0)
        ppy = intr.get('ppy', 0)
        w, h = 640, 480
        scale_x = w / (ppx * 2) if ppx > 0 else 1.0
        scale_y = h / (ppy * 2) if ppy > 0 else 1.0

        intrinsics = {
            'fx': intr['fx'] * scale_x,
            'fy': intr['fy'] * scale_y,
            'cx': w / 2.0,
            'cy': h / 2.0,
        }

        distortion = {
            'model': intr.get('distortion_model', 'plumb_bob'),
            'coeffs': [
                intr.get('k1', 0), intr.get('k2', 0),
                intr.get('p1', 0), intr.get('p2', 0),
                intr.get('k3', 0),
            ],
        }

        # --- Extrinsic ---
        aligned_path = params_dir / "head_extrinsic_params_aligned.json"
        static_path = params_dir / "head_extrinsic_params.json"

        if aligned_path.exists():
            with open(aligned_path) as f:
                aligned = json.load(f)
            # Clamp frame index to valid range
            idx = min(obs_original_frame, len(aligned) - 1)
            ext = aligned[idx]['extrinsic']
        elif static_path.exists():
            with open(static_path) as f:
                ext = json.load(f)['extrinsic']
        else:
            raise FileNotFoundError(
                f"No extrinsic file found in {params_dir}"
            )

        R = np.array(ext['rotation_matrix'], dtype=np.float64)
        t = np.array(ext['translation_vector'], dtype=np.float64)
        T = _make_T_world_to_cam(R, t)

        return intrinsics, T, distortion
