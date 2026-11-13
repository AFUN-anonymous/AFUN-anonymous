"""Shared data classes for dataset preprocessing."""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class CameraInfo:
    """Camera calibration and metadata."""
    cam_id: str                          # unique camera identifier (e.g. serial number)
    intrinsics: Dict[str, Any]           # {fx, fy, cx, cy, ...}
    T_base_to_cam: np.ndarray            # 4x4 base(world)-to-camera transform
    extrinsic_source: str                # provenance of extrinsics
    intrinsic_source: str                # provenance of intrinsics
    distortion: Optional[Dict[str, Any]] = None


@dataclass
class Interval:
    """Manipulation interval [start, end], as frame indices into the episode.

    ``obs_frame`` and ``contact_frame`` default to ``start`` when left as None.
    ``camera_infos`` overrides ``EpisodeData.camera_infos`` for this interval.
    """
    start: int
    end: int
    language: Optional[str] = None
    obs_frame: Optional[int] = None
    contact_frame: Optional[int] = None
    camera_infos: Optional[List[CameraInfo]] = None

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory."""
    frame_idx: int
    position_3d: List[float]             # [x, y, z] in camera frame, meters
    position_pixel: Optional[List[int]] = None  # [u, v] in image


@dataclass
class Motion:
    """Motion descriptor (first ~1 s of interval)."""
    motion_3d: Optional[List[float]] = None      # normalized 3D displacement direction
    motion_2d_start: Optional[List[int]] = None  # start pixel of 2D motion
    motion_2d_end: Optional[List[int]] = None    # end pixel of 2D motion


@dataclass
class EpisodeData:
    """Everything load_episode() returns for one episode.

    ``extra`` holds dataset-specific state used by the adapter; it is not saved.
    """
    num_frames: int
    fps: float
    camera_infos: List[CameraInfo]
    extra: Dict[str, Any] = field(default_factory=dict)
