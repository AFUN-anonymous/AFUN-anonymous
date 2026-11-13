"""
RoboMIND 2.0 preprocessing (evaluation only). Inherits from RoboDatasetBase.

The robot family is read from the first path component under ``root``::

    {root}/{top_dir}/data/{robot}/{task}/success_episodes/{ep_id}/data/trajectory.hdf5
    top_dir: Tienkung -> tienkung, 1/2/3 -> franka, UR5 -> ur5, Ark -> ark

| Field                                         | Tienkung      | Franka (1/2/3) | UR5           | Ark           |
|-----------------------------------------------|---------------|----------------|---------------|---------------|
| puppet/arm_{l,r}_position_align shape         | (N, 7)        | (N, 8)         | (N, 6)        | (N, 6)        |
| puppet/end_effector_{l,r}_position_align      | (N, 1) [0,1]  | (N, 1) [0,1]   | (N, 1) [0,1]  | (N, 1) [-32,-6] or [0,5] |
| puppet/end_effector_{l,r}_pose_align          | (0,) empty    | (N, 7) meters  | (N, 7) meters | (N, 7) near-zero |
| gripper threshold (closed below)              | 0.5           | 0.5            | 0.5           | auto (-20 or 2.5) |
| static cameras                                | camera_top    | front/l/r/top  | l/r           | l/r/top       |
| get_gripper_tip_3d                            | zeros         | midpoint EE xyz| midpoint EE xyz| zeros        |
| episode_id prefix                             | robomind2_tienkung_ | robomind2_franka_{N}_ | robomind2_ur5_ | robomind2_ark_ |

Intervals are the spans where min(left, right) gripper state is below the
family threshold; the whole episode is used when none is found. Non-rigid tasks
are skipped. Extrinsics are not available, so T_base_to_cam is identity.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import CameraInfo, EpisodeData, Interval

log = logging.getLogger(__name__)

# Strict (family, cam_id) -> 3x3 K table. The single source of truth for
# which RoboMIND2 cameras enter the pipeline.
#
# Values come from the official intrinsics document
# (https://x-humanoid-robomind.github.io/static/RoboMIND_intrinsics.md).
# Mapping:
#   tienkung                  -> TienKung 1RGB
#   franka_{1,2,3} (4 cams)   -> Franka Fr3 Dual Arm (left/right/top/front)
#   ur5 (left, right only)    -> UR_1RGB; ur5/{top,front} have no published
#                                K and are intentionally absent
#   ark/{left,right}          -> AgileX_3RGB.{left,right}
#   ark/camera_top            -> AgileX_3RGB.front (same hardware, renamed)
# Wrist cameras also have no published K and are intentionally absent.
#
# A (family, cam_id) missing from this table causes the camera to be
# dropped at load_episode time; _get_intrinsics raises if reached with an
# unknown key.
#
# Resolution note: for some (family, cam_id) the published K was calibrated
# at a resolution that differs from the RGB stream stored in the hdf5
# (e.g. the franka camera_left K is for 480x640 but some episodes record at
# 1280x720). K is stored here as published; downstream callers that need
# K consistent with image dimensions should rescale based on the actual
# H/W of the loaded frame.
ROBOMIND2_INTRINSICS: Dict[Tuple[str, str], np.ndarray] = {
    # TienKung 1RGB -- published K is for 1280x720 (cx~645) but the hdf5 stores
    # 640x480 (downsampled by 0.5). Scale fx/fy/cx/cy by 0.5 to match the
    # actual stored resolution; bottom-right corner stays 1.
    # (Other families' K already match their hdf5 resolutions; tienkung is the
    # only outlier.)
    ("tienkung", "camera_top"):   np.array([
        [687.6524658203125 / 2,    0.0,                       645.3302612304688 / 2],
        [0.0,                      687.55224609375 / 2,       361.4851989746094 / 2],
        [0.0,                      0.0,                       1.0],
    ]),
    # Franka Fr3 Dual Arm  (left, right at 640x480; top, front at 1280x720)
    ("franka",   "camera_left"):  np.array([
        [462.0193176269531, 0.0,                317.4200744628906],
        [0.0,               462.0207214355469,  239.2024688720703],
        [0.0,               0.0,                1.0],
    ]),
    ("franka",   "camera_right"): np.array([
        [462.4823303222656, 0.0,                321.53204345703125],
        [0.0,               462.3477783203125,  240.60427856445312],
        [0.0,               0.0,                1.0],
    ]),
    ("franka",   "camera_top"):   np.array([
        [693.0001220703125, 0.0,                642.3272094726562],
        [0.0,               692.7867431640625,  358.94342041015625],
        [0.0,               0.0,                1.0],
    ]),
    ("franka",   "camera_front"): np.array([
        [612.8556518554688, 0.0,                641.6568603515625],
        [0.0,               612.7535400390625,  347.0881042480469],
        [0.0,               0.0,                1.0],
    ]),
    # UR_1RGB  (only the 640x480 left + right; ur5/{top,front} INTENTIONALLY OMITTED)
    ("ur5",      "camera_left"):  np.array([
        [609.3246327846327, 0.0,                317.7893242938894],
        [0.0,               609.7463827932434,  257.3246343287439],
        [0.0,               0.0,                1.0],
    ]),
    ("ur5",      "camera_right"): np.array([
        [609.3246327846327, 0.0,                317.7893242938894],
        [0.0,               609.7463827932434,  257.3246343287439],
        [0.0,               0.0,                1.0],
    ]),
    # AgileX_3RGB -> ark
    ("ark",      "camera_left"):  np.array([
        [454.6264953613281, 0.0,                327.5919494628906],
        [0.0,               454.6264953613281,  238.47760009765625],
        [0.0,               0.0,                1.0],
    ]),
    ("ark",      "camera_right"): np.array([
        [454.0530700683594, 0.0,                327.8827209472656],
        [0.0,               454.0530700683594,  238.57809448242188],
        [0.0,               0.0,                1.0],
    ]),
    # AgileX 'front' renamed to ark 'camera_top' in RoboMIND2 -- same hardware
    ("ark",      "camera_top"):   np.array([
        [455.47412109375,   0.0,                327.1614074707031],
        [0.0,               455.47412109375,    236.7096710205078],
        [0.0,               0.0,                1.0],
    ]),
}

# Task-name substrings indicating non-rigid object manipulation.
NON_RIGID_TASK_KEYWORDS = {
    "cloth", "towel", "fold", "fabric", "rope", "string",
    "tape", "cable", "sweep", "garbage",
}

# Ark gripper signal auto-detection thresholds.
#   Encoding A (2024-04~06): values in [-32, -6]; threshold = -20.0
#   Encoding B (2024-07~08): values in [0, 5];    threshold =   2.5
# Both encodings share closed_when="below" semantics.
_ARK_ENCODING_THRESHOLDS = {"A": -20.0, "B": 2.5}

# Maps each top-level directory name (lowercased) to its robot family.
_TOP_DIR_TO_FAMILY = {
    "tienkung": "tienkung",
    "1": "franka",
    "2": "franka",
    "3": "franka",
    "ark": "ark",
    "ur5": "ur5",
}

# Per-family gripper detection parameters (threshold, closed_when).
# Ark uses a very different negative range (-32..-6).
_FAMILY_GRIPPER_PARAMS: Dict[str, Dict[str, Any]] = {
    "tienkung": {"threshold": 0.5, "closed_when": "below"},
    "franka":   {"threshold": 0.5, "closed_when": "below"},
    "ur5":      {"threshold": 0.5, "closed_when": "below"},
    "ark":      {"threshold": -20.0, "closed_when": "below"},
}

def _safe_name(value: str) -> str:
    """Convert a string to a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _task_to_language(task_name: str) -> str:
    """Derive a natural-language instruction from a task folder name."""
    return task_name.replace("_", " ")


def _is_non_rigid_task(task_name: str) -> bool:
    """Check if a task involves non-rigid objects."""
    lower = task_name.lower()
    return any(kw in lower for kw in NON_RIGID_TASK_KEYWORDS)


def _detect_closed_intervals(
    values: np.ndarray,
    threshold: float,
    min_len: int,
    closed_when: str,
) -> List[Interval]:
    """Detect contiguous intervals where signal crosses threshold."""
    if values.size == 0:
        return []
    if closed_when == "below":
        closed = values < threshold
    else:
        closed = values > threshold

    intervals: List[Interval] = []
    start: Optional[int] = None
    for idx, is_closed in enumerate(closed):
        if is_closed and start is None:
            start = idx
        elif not is_closed and start is not None:
            if idx - start >= min_len:
                intervals.append(Interval(start=start, end=idx - 1))
            start = None
    if start is not None and len(closed) - start >= min_len:
        intervals.append(Interval(start=start, end=len(closed) - 1))
    return intervals


class RoboMIND2Preprocess(RoboDatasetBase):
    """RoboMIND 2.0 dataset preprocessor supporting Tienkung, Franka, UR5, and Ark."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.root = Path(dataset.root)
        self._extract_depth = bool(getattr(dataset, "extract_depth", True))
        self.default_fps = float(getattr(dataset, "default_fps", 30.0))
        self.min_interval_time = float(getattr(dataset, "min_interval_time", 0.5))

    # ------------------------------------------------------------------
    # Family detection helpers
    # ------------------------------------------------------------------

    def _detect_family(self, episode_path: str) -> str:
        """Return the robot family for a given episode path.

        Returns one of ``'tienkung'``, ``'franka'``, ``'ark'``, ``'ur5'``,
        or ``'unknown'`` when the top-level directory is not a known family.
        """
        p = Path(episode_path)
        rel = p.relative_to(self.root)
        top = rel.parts[0].lower()

        if top in _TOP_DIR_TO_FAMILY:
            return _TOP_DIR_TO_FAMILY[top]

        log.warning(
            "Unknown top-level directory '%s' in episode path %s -- "
            "returning family='unknown'. Update _TOP_DIR_TO_FAMILY to "
            "support this family.",
            rel.parts[0], episode_path,
        )
        return "unknown"

    def _parse_path_parts(self, episode_path: str) -> Tuple[str, str, str]:
        """Return (family, task_name, ep_id) from an episode path.

        parts = (top_dir, 'data', robot_type, task_name, 'success_episodes', ep_id, ...)
        """
        p = Path(episode_path)
        rel = p.relative_to(self.root)
        parts = rel.parts
        family = self._detect_family(episode_path)
        task_name = parts[3] if len(parts) > 3 else "unknown"
        ep_id = parts[5] if len(parts) > 5 else "unknown"
        # For Franka: top_dir is '1'/'2'/'3' -- embedded in the id.
        return family, task_name, ep_id

    def _franka_dir_idx(self, episode_path: str) -> str:
        """Return the Franka directory index ('1', '2', or '3')."""
        p = Path(episode_path)
        rel = p.relative_to(self.root)
        return rel.parts[0]  # '1', '2', or '3'

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def discover_episodes(self) -> List[str]:
        """Find all trajectory.hdf5 files under the dataset root."""
        episodes = []
        for hdf5_path in sorted(self.root.rglob("trajectory.hdf5")):
            path_str = str(hdf5_path)

            # Skip macOS AppleDouble artifacts
            if "/._____temp/" in path_str or any(
                part.startswith("._") for part in hdf5_path.parts
            ):
                log.debug("Skipping artifact path: %s", hdf5_path)
                continue

            # Determine task name for this path
            try:
                _family, task_name, _ep_id = self._parse_path_parts(path_str)
            except (ValueError, IndexError):
                log.warning("Could not parse path: %s -- skipping", hdf5_path)
                continue

            # Skip episodes from unknown robot families
            if _family == "unknown":
                log.warning(
                    "Skipping episode with unknown robot family: %s", hdf5_path
                )
                continue

            # Skip non-rigid tasks
            if _is_non_rigid_task(task_name):
                log.info("Skipping non-rigid task: %s", task_name)
                continue

            episodes.append(path_str)

        log.info("Discovered %d episodes under %s", len(episodes), self.root)
        return episodes

    def get_episode_id(self, episode_path: str) -> str:
        """Generate a unique filesystem-safe episode ID."""
        family, task_name, ep_id = self._parse_path_parts(episode_path)
        task_safe = _safe_name(task_name)
        ep_safe = _safe_name(ep_id)
        if family == "franka":
            dir_idx = self._franka_dir_idx(episode_path)
            return f"robomind2_franka_{dir_idx}_{task_safe}_{ep_safe}"
        elif family == "ark":
            return f"robomind2_ark_{task_safe}_{ep_safe}"
        elif family == "ur5":
            return f"robomind2_ur5_{task_safe}_{ep_safe}"
        else:  # tienkung
            return f"robomind2_tienkung_{task_safe}_{ep_safe}"

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load a single trajectory HDF5 and return EpisodeData."""
        p = Path(episode_path)
        h5 = h5py.File(p, "r")

        # Validate required HDF5 groups
        if "camera_observations/color_images" not in h5:
            h5.close()
            raise ValueError(
                f"Missing camera_observations/color_images in {episode_path}"
            )

        family = self._detect_family(episode_path)
        _, task_name, _ = self._parse_path_parts(episode_path)

        # Discover all available cameras from color_images group
        color_group = h5["camera_observations/color_images"]
        all_cam_names = sorted(color_group.keys())
        if not all_cam_names:
            h5.close()
            raise ValueError(f"No cameras found in {episode_path}")

        # K-table membership is the single source of truth for which cams enter
        # the pipeline. Wrist cams and ur5/{top,front} are intentionally absent
        # from ROBOMIND2_INTRINSICS, so this one check covers all skip cases.
        cam_names = [c for c in all_cam_names if (family, c) in ROBOMIND2_INTRINSICS]
        if not cam_names:
            h5.close()
            raise ValueError(
                f"No cameras with published K for family={family!r} in {episode_path}. "
                f"Available: {all_cam_names}. Update ROBOMIND2_INTRINSICS if this is a new family."
            )

        num_frames = color_group[all_cam_names[0]].shape[0]
        if num_frames <= 0:
            h5.close()
            raise ValueError(f"Zero-frame episode: {episode_path}")

        # Compute FPS from timestamps
        ts_key = "camera_observations/timestamp"
        if ts_key in h5:
            timestamps = h5[ts_key][:]
        else:
            timestamps = np.array([])

        if len(timestamps) > 1 and timestamps[-1] > timestamps[0]:
            duration_s = float(timestamps[-1] - timestamps[0])
            fps = num_frames / duration_s
            if fps > 120 or fps < 1:
                log.warning(
                    "Computed FPS=%.1f outside expected range for %s, using default",
                    fps, episode_path,
                )
                fps = self.default_fps
        else:
            fps = self.default_fps
            log.warning(
                "Could not compute FPS from timestamps for %s, using default %.1f",
                episode_path, fps,
            )

        # Build CameraInfo for each static camera
        camera_infos = []
        for cam_id in cam_names:
            intrinsics = self._get_intrinsics(family, cam_id)
            T_base_to_cam = np.eye(4, dtype=np.float64)
            camera_infos.append(
                CameraInfo(
                    cam_id=cam_id,
                    intrinsics=intrinsics,
                    T_base_to_cam=T_base_to_cam,
                    extrinsic_source="identity (static camera, no extrinsics available)",
                    intrinsic_source="RoboMIND 2.0 published intrinsics",
                )
            )

        # Load gripper states for interval detection (family-specific)
        gripper_params = _FAMILY_GRIPPER_PARAMS[family]
        gripper_left = self._load_gripper_state(
            h5, "left", num_frames, family=family
        )
        gripper_right = self._load_gripper_state(
            h5, "right", num_frames, family=family
        )

        # For Ark episodes, auto-detect the gripper encoding from the signal range
        # because two distinct encodings are mixed across recording dates:
        #   Encoding A (2024-04~06): values in [-32, -6]; threshold = _ARK_ENCODING_THRESHOLDS["A"] = -20.0
        #   Encoding B (2024-07~08): values in [0, 5];    threshold = _ARK_ENCODING_THRESHOLDS["B"] =   2.5
        # Both encodings share closed_when="below" semantics.
        if family == "ark":
            # Collect all available gripper values to inspect the range.
            all_gripper_vals = np.concatenate(
                [v for v in (gripper_left, gripper_right) if v is not None]
            ) if any(v is not None for v in (gripper_left, gripper_right)) else np.array([])
            if all_gripper_vals.size == 0:
                # No gripper data at all -- default to Encoding A threshold and warn.
                ark_threshold = _ARK_ENCODING_THRESHOLDS["A"]
                log.warning(
                    "Ark gripper signal is empty for episode %s; "
                    "defaulting to Encoding A threshold=%.1f. "
                    "Interval detection may produce zero intervals.",
                    episode_path, ark_threshold,
                )
            elif np.nanmax(all_gripper_vals) >= 0:
                # Positive values present -> Encoding B
                ark_threshold = _ARK_ENCODING_THRESHOLDS["B"]
            else:
                # All-negative -> Encoding A
                ark_threshold = _ARK_ENCODING_THRESHOLDS["A"]
            log.debug(
                "Ark gripper auto-detect: max=%.3f -> threshold=%.1f for %s",
                float(np.nanmax(all_gripper_vals)) if all_gripper_vals.size > 0 else float("nan"),
                ark_threshold,
                episode_path,
            )
            gripper_threshold = ark_threshold
        else:
            gripper_threshold = gripper_params["threshold"]

        # Load EE pose for families that have it (Franka, UR5)
        ee_pose_left = self._load_ee_pose(h5, "left", family)
        ee_pose_right = self._load_ee_pose(h5, "right", family)

        return EpisodeData(
            num_frames=num_frames,
            fps=fps,
            camera_infos=camera_infos,
            extra={
                "h5": h5,
                "family": family,
                "task_name": task_name,
                "language": _task_to_language(task_name),
                "gripper_left": gripper_left,
                "gripper_right": gripper_right,
                "gripper_threshold": gripper_threshold,
                "gripper_closed_when": gripper_params["closed_when"],
                "ee_pose_left": ee_pose_left,
                "ee_pose_right": ee_pose_right,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Detect manipulation intervals from gripper closure."""
        signal = self._gripper_signal(episode_data)
        if signal is None:
            return [
                Interval(
                    start=0,
                    end=episode_data.num_frames - 1,
                    language=episode_data.extra.get("language"),
                    obs_frame=0,
                    contact_frame=0,
                )
            ]

        min_len = max(1, int(self.min_interval_time * episode_data.fps))
        intervals = _detect_closed_intervals(
            signal,
            episode_data.extra["gripper_threshold"],
            min_len,
            episode_data.extra["gripper_closed_when"],
        )

        language = episode_data.extra.get("language")
        for i, iv in enumerate(intervals):
            iv.language = language
            iv.contact_frame = iv.start
            if i == 0:
                iv.obs_frame = 0
            else:
                iv.obs_frame = intervals[i - 1].end + 1

        if not intervals:
            log.warning(
                "No intervals detected for task=%s, using whole episode",
                episode_data.extra.get("task_name"),
            )
            intervals = [
                Interval(
                    start=0,
                    end=episode_data.num_frames - 1,
                    language=language,
                    obs_frame=0,
                    contact_frame=0,
                )
            ]

        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Decode RGB and depth for a given camera and frame."""
        h5 = episode_data.extra["h5"]

        # Decode RGB (stored as BGR JPEG)
        rgb_raw = h5[f"camera_observations/color_images/{cam_id}"][frame_idx]
        if isinstance(rgb_raw, bytes):
            rgb_bytes = np.frombuffer(rgb_raw, np.uint8)
        else:
            rgb_bytes = np.asarray(rgb_raw, dtype=np.uint8).ravel()
        bgr = cv2.imdecode(rgb_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(
                f"Failed to decode RGB frame {frame_idx} for camera {cam_id}"
            )
        # RoboMIND 2.0 hdf5 stores frames in BGR channel order; convert to RGB.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Decode depth
        depth = None
        if self._extract_depth:
            depth_key = f"camera_observations/depth_images/{cam_id}"
            if depth_key in h5:
                depth_raw = h5[depth_key][frame_idx]
                if isinstance(depth_raw, bytes):
                    depth_bytes = np.frombuffer(depth_raw, np.uint8)
                else:
                    depth_bytes = np.asarray(depth_raw, dtype=np.uint8).ravel()
                depth_decoded = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)
                if depth_decoded is not None:
                    # Depth is uint16 in millimeters -- cast to float32
                    depth = depth_decoded.astype(np.float32)
                else:
                    log.warning(
                        "Failed to decode depth frame %d for camera %s",
                        frame_idx, cam_id,
                    )

        return rgb, depth

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Return 3D gripper tip position in robot base frame (meters).

        - Tienkung / Ark: no valid Cartesian EE data -> zeros.
        - Franka / UR5: midpoint of left+right EE positions from puppet pose.
          Falls back to available arm or zeros if both invalid.
        """
        family = episode_data.extra.get("family", "tienkung")
        if family not in ("franka", "ur5"):
            return np.zeros(3, dtype=np.float64)

        ee_left = episode_data.extra.get("ee_pose_left")
        ee_right = episode_data.extra.get("ee_pose_right")

        pos_left = None
        pos_right = None

        if ee_left is not None and frame_idx < len(ee_left):
            row = ee_left[frame_idx]
            if np.isfinite(row[:3]).all():
                pos_left = row[:3].astype(np.float64)

        if ee_right is not None and frame_idx < len(ee_right):
            row = ee_right[frame_idx]
            if np.isfinite(row[:3]).all():
                pos_right = row[:3].astype(np.float64)

        if pos_left is not None and pos_right is not None:
            return (pos_left + pos_right) / 2.0
        if pos_left is not None:
            return pos_left
        if pos_right is not None:
            return pos_right
        return np.zeros(3, dtype=np.float64)

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Return dummy zeros -- flange data not available in RoboMIND2."""
        return np.zeros(3, dtype=np.float64)

    def close_episode(self, episode_data: EpisodeData) -> None:
        """Close the HDF5 file handle."""
        h5 = episode_data.extra.pop("h5", None)
        if h5 is not None:
            h5.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gripper_signal(episode_data: EpisodeData) -> Optional[np.ndarray]:
        """Element-wise min of the left/right gripper states (either arm closing counts)."""
        gripper_left = episode_data.extra.get("gripper_left")
        gripper_right = episode_data.extra.get("gripper_right")
        if gripper_left is None and gripper_right is None:
            return None

        if gripper_left is not None and gripper_right is not None:
            return np.minimum(gripper_left, gripper_right)
        if gripper_left is not None:
            return gripper_left
        return gripper_right

    def _get_intrinsics(
        self, family: str, cam_id: str
    ) -> Dict[str, float]:
        """Return camera K from the strict (family, cam_id) lookup table.

        No fallback / no estimation. ``cam_names`` should have already filtered
        out any (family, cam_id) not in ``ROBOMIND2_INTRINSICS``; reaching this
        function with an unknown key is therefore a programmer error.
        """
        K = ROBOMIND2_INTRINSICS.get((family, cam_id))
        if K is None:
            raise ValueError(
                f"No published K for ({family!r}, {cam_id!r}). "
                f"Adapter contract: load_episode's cam_names filter should "
                f"have dropped this. Update ROBOMIND2_INTRINSICS coverage."
            )
        return {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }

    @staticmethod
    def _load_gripper_state(
        h5: h5py.File, side: str, num_frames: int, family: str = "tienkung"
    ) -> Optional[np.ndarray]:
        """Load gripper state for one arm.

        Returns a 1D float32 array of length ``num_frames``, or None if not found.

        Values may be outside [0, 1]; the Ark family encodes gripper state in
        [-32, -6] and must pass through unchanged.

        Family-specific prefix policy:
          - ``ark``: only try ``puppet/...``; Ark recordings have no master
            gripper signal.
          - all others: try ``master`` first (operator-side normalised
            command, 0=closed, 1=open), then ``puppet`` as fallback.
        """
        prefixes = ("puppet",) if family == "ark" else ("master", "puppet")
        for prefix in prefixes:
            key = f"{prefix}/end_effector_{side}_position_align/data"
            if key not in h5:
                continue
            data = h5[key][:]
            if data.size == 0:
                continue
            # (N, 1) -> scalar gripper state; (N, M) with M>1 -> dexterous hand joints
            if data.ndim == 2 and data.shape[1] == 1:
                data = data[:, 0]
            elif data.ndim == 2 and data.shape[1] > 1:
                log.debug(
                    "Skipping %s: multi-DOF hand data shape=%s",
                    key, data.shape,
                )
                continue
            # Guard against (0, 1) -> (0,) after reshape -- linspace(0, -1, N) otherwise.
            if len(data) == 0:
                continue
            # Align frame counts if needed
            if len(data) != num_frames:
                indices = np.linspace(0, len(data) - 1, num_frames).astype(int)
                data = data[indices]
            return data
        return None

    @staticmethod
    def _load_ee_pose(
        h5: h5py.File, side: str, family: str
    ) -> Optional[np.ndarray]:
        """Load Cartesian EE pose (N, 7) for families that have it.

        Returns (N, 7) float64 array or None.  Only Franka and UR5 have
        valid (non-near-zero) Cartesian EE data.
        """
        if family not in ("franka", "ur5"):
            return None
        key = f"puppet/end_effector_{side}_pose_align/data"
        if key not in h5:
            return None
        data = h5[key][:]
        if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
            return None
        return data.astype(np.float64)
