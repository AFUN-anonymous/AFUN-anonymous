"""
Template for a new dataset adapter.

Copy this file, rename the class, and fill in every method marked TODO. Then add
config/dataset/<name>.yaml with ``_target_: datasets.<File>.<Class>`` and
``name: <name>`` (the output folder under output_root), and run
``python preprocess.py <name>``.

Conventions (see preprocess/base.py):
  - depth: float32 millimeters
  - 3D positions: meters
  - extrinsics: T_base_to_cam (4x4, robot base/world -> camera)
"""

import numpy as np
from typing import List, Tuple, Optional

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, EpisodeData


class TemplatePreprocess(RoboDatasetBase):
    """Adapter template. Every abstract method of RoboDatasetBase is stubbed."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # TODO: read dataset options from cfg.dataset (e.g. self.root = cfg.dataset.root)
        #   and pre-load resources shared across episodes (calibration, annotations).

    def discover_episodes(self) -> List[str]:
        """Return every episode path to process (used when episode_txt_list is null).

        TODO: e.g. sorted(str(p.parent) for p in Path(root).rglob("trajectory.h5")).
        """
        raise NotImplementedError("discover_episodes")

    def get_episode_id(self, episode_path: str) -> str:
        """Return a filesystem-safe ID, unique within and across datasets.

        TODO: e.g. a UUID from metadata or the relative path with "/" -> "_".
        """
        raise NotImplementedError("get_episode_id")

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load one episode.

        TODO:
          1. Open the episode files and read the robot state.
          2. Build one CameraInfo per static camera view
             (cam_id, intrinsics {fx, fy, cx, cy}, T_base_to_cam, provenance strings).
          3. Return EpisodeData(num_frames=..., fps=..., camera_infos=[...], extra={...});
             ``extra`` holds whatever the other methods need and is not saved.
        """
        raise NotImplementedError("load_episode")

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Return the manipulation intervals.

        TODO: e.g. gripper close/open spans or annotated segments. For each
        Interval set start/end (inclusive frame indices), and optionally
        obs_frame and contact_frame (both default to start) and language.
        """
        raise NotImplementedError("detect_intervals")

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (RGB uint8 HxWx3, depth float32 HxW in mm, or None).

        TODO: read from the dataset's video, image folder or HDF5.
        """
        raise NotImplementedError("extract_frame_and_depth")

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Return the gripper tip (3,) in the robot base frame, meters.

        TODO: e.g. flange position + tip offset along the flange z-axis.
        """
        raise NotImplementedError("get_gripper_tip_3d")

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Return the flange (3,) in the robot base frame, meters (no tip offset)."""
        raise NotImplementedError("get_flange_3d")

    def close_episode(self, episode_data: EpisodeData) -> None:
        """TODO: close file handles / video readers stored in episode_data.extra."""
        pass

    # Optional overrides:
    #   extract_frame()                       RGB-only fast path (used by mainprocess tracking)
    #   compute_trajectory()                  when the tip is not in the base frame
    #   reproject_camera_point_to_obs_frame() required for moving cameras
