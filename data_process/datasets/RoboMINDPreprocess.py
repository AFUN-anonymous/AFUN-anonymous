"""
RoboMIND (v1) preprocessing.

Each episode is one ``trajectory.hdf5`` under
``{root}/<embodiment>/<task>/success_episodes/<split>/<episode>/data/``.
Embodiments (Franka, AgileX, UR, Tien Kung, simulation) differ in camera names,
color order, and where the end-effector and gripper states are stored; see the
per-embodiment helpers below.

- Cameras: arm-mounted views (MOVING_CAMERAS) are dropped. Intrinsics come from
  the published RoboMIND_intrinsics.md (image-size fallback otherwise).
  Extrinsics are not published, so T_base_to_cam is identity and 2D
  trajectory pixels are rescaled to fit the image (compute_trajectory).
- Depth: decoded from the HDF5 and converted to mm when the values look like
  meters (max <= 100).
- Intervals: the per-step language annotations (annotation_json_dir) give
  frame boundaries for each step; the whole episode is one interval when no
  annotation is found.
- Non-rigid tasks (cloth, rope, ...) are skipped at discovery.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import yaml

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import CameraInfo, EpisodeData, Interval, TrajectoryPoint

log = logging.getLogger(__name__)


BGR_EMBODIMENTS = {
    "h5_franka_3rgb",
    "h5_franka_1rgb",
    "h5_ur_1rgb",
    "h5_franka_fr3_dual",
}

DOC_INTRINSICS_KEYS = {
    "h5_agilex_3rgb": "AgileX_3RGB",
    "h5_franka_3rgb": "Franka_3RGB",
    "h5_franka_1rgb": "Franka_1RGB",
    "h5_tienkung_gello_1rgb": "TienKung",
    "h5_tienkung_xsens_1rgb": "TienKung",
    "h5_ur_1rgb": "UR_1RGB",
}

# Cameras mounted on the robot arm -- must be excluded (pipeline assumes static cameras).
MOVING_CAMERAS: Dict[str, set] = {
    "h5_agilex_3rgb": {"camera_left_wrist", "camera_right_wrist"},
    "h5_simulation": {"camera_handeye"},
}

# Task-name substrings indicating non-rigid object manipulation.
# The 3D trajectory and mask tracking pipeline assumes rigid bodies.
NON_RIGID_TASK_KEYWORDS = {
    "cloth", "towel", "fold", "wipe_cloth", "fabric", "rope", "string",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _parse_metadata(root_dir: Path, trajectory_path: Path) -> Dict[str, Optional[str]]:
    rel = trajectory_path.relative_to(root_dir)
    parts = rel.parts
    metadata = {
        "dataset_name": parts[0] if len(parts) > 0 else None,
        "task_name": None,
        "split": None,
        "episode_id": None,
    }
    if len(parts) >= 7 and parts[2] == "success_episodes":
        metadata["task_name"] = parts[1]
        metadata["split"] = parts[3]
        metadata["episode_id"] = parts[4]
    elif len(parts) >= 5 and parts[1] == "success_episodes":
        metadata["task_name"] = parts[0]
        metadata["episode_id"] = parts[2]
    elif len(parts) >= 3:
        metadata["episode_id"] = parts[-3]
    return metadata


def _is_compressed_frame(frame: Any) -> bool:
    return isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 1


def _decode_frame(frame: Any, embodiment: str, sensor_name: str) -> np.ndarray:
    if _is_compressed_frame(frame):
        read_flag = cv2.IMREAD_UNCHANGED if sensor_name == "depth_images" else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(frame, read_flag)
        if decoded is None:
            raise ValueError(f"Failed to decode compressed {sensor_name} frame.")
        image = decoded
    else:
        image = np.asarray(frame)

    if sensor_name == "rgb_images" and embodiment in BGR_EMBODIMENTS:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _matrix_to_intrinsics_dict(K: np.ndarray) -> Dict[str, float]:
    K = np.asarray(K, dtype=np.float64)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def _intrinsics_camera_aliases(cam_id: str) -> List[str]:
    aliases = [cam_id]
    if cam_id == "camera_left_wrist":
        aliases.append("camera_left")
    elif cam_id == "camera_right_wrist":
        aliases.append("camera_right")
    return aliases


class RoboMINDPreprocess(RoboDatasetBase):
    """RoboMIND dataset preprocessor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.root = Path(dataset.root)
        self._extract_depth = bool(getattr(dataset, "extract_depth", True))
        self.default_fps = float(getattr(dataset, "default_fps", 10.0))
        self.min_interval_time = float(getattr(dataset, "min_interval_time", 0.5))

        self.schema_map = self._load_schema_map(Path(dataset.schema_path))
        self.intrinsics_map = self._load_intrinsics_map(Path(dataset.intrinsics_path))
        # Per-step language annotations: annotation_id -> {task_summary, steps[{step_description, start_frame, end_frame}]}
        ann_dir = getattr(dataset, "annotation_json_dir", None)
        self.step_annotations = self._load_step_annotations(ann_dir)

    @staticmethod
    def _parse_frame_number(frame_str: str) -> int:
        """Extract frame index from annotation filename like 'camera_top_0038.jpg'."""
        m = re.search(r"_(\d+)\.jpg$", frame_str)
        if m:
            return int(m.group(1))
        return 0

    @staticmethod
    def _load_step_annotations(ann_dir: Optional[str]) -> Dict[str, dict]:
        """Load per-step language annotations from JSON files.

        Returns a dict keyed by annotation ID (matches the relative path
        from dataset root to the ``data/`` directory, e.g.
        ``h5_agilex_3rgb/14_steamcorn/success_episodes/train/xxx/data``).
        """
        if ann_dir is None:
            return {}
        ann_path = Path(ann_dir)
        if not ann_path.is_dir():
            log.warning("annotation_json_dir not found: %s", ann_path)
            return {}
        result: Dict[str, dict] = {}
        for json_file in sorted(ann_path.glob("*.json")):
            try:
                entries = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning("Failed to load %s: %s", json_file, e)
                continue
            for entry in entries:
                ann_id = entry.get("id", "")
                response = entry.get("response", {})
                if response and response.get("steps"):
                    result[ann_id] = response
        log.info("Loaded %d step annotations from %s", len(result), ann_path)
        return result

    @staticmethod
    def _load_schema_map(schema_path: Path) -> Dict[str, Any]:
        with open(schema_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("domain_dict", {})

    @staticmethod
    def _load_intrinsics_map(intrinsics_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
        with open(intrinsics_path, "r", encoding="utf-8") as f:
            text = f.read().replace("：", ":")  # full-width colon -> ASCII

        sections: Dict[str, Any] = {}
        for match in re.finditer(r"^## .+?$([\s\S]*?)(?=^## |\Z)", text, re.MULTILINE):
            block = match.group(1).strip()
            if not block:
                continue
            loaded = yaml.safe_load(block)
            if isinstance(loaded, dict):
                sections.update(loaded)

        parsed: Dict[str, Dict[str, Dict[str, float]]] = {}
        for embodiment, doc_key in DOC_INTRINSICS_KEYS.items():
            section = sections.get(doc_key)
            if section is None:
                continue
            if "intrinsics" in section:
                parsed[embodiment] = {
                    "camera_top": _matrix_to_intrinsics_dict(np.asarray(section["intrinsics"]))
                }
                continue

            camera_map: Dict[str, Dict[str, float]] = {}
            for short_name, value in section.items():
                camera_name = f"camera_{short_name}"
                camera_map[camera_name] = _matrix_to_intrinsics_dict(
                    np.asarray(value["intrinsics"])
                )
            parsed[embodiment] = camera_map
        return parsed

    @staticmethod
    def _is_non_rigid_task(task_name: str) -> bool:
        task_lower = task_name.lower()
        return any(kw in task_lower for kw in NON_RIGID_TASK_KEYWORDS)

    def discover_episodes(self) -> List[str]:
        episodes: List[str] = []
        for trajectory_path in sorted(self.root.rglob("trajectory.hdf5")):
            if "success_episodes" not in trajectory_path.parts:
                continue
            metadata = _parse_metadata(self.root, trajectory_path)
            dataset_name = metadata["dataset_name"] or ""
            task_name = metadata["task_name"] or dataset_name
            # Skip non-rigid tasks (cloth, rope, etc.)
            if self._is_non_rigid_task(task_name):
                continue
            episodes.append(str(trajectory_path))
        return episodes

    def get_episode_id(self, episode_path: str) -> str:
        meta = _parse_metadata(self.root, Path(episode_path))
        embodiment = meta["dataset_name"] or "robomind"
        task_name = meta["task_name"] or embodiment
        episode_id = meta["episode_id"] or Path(episode_path).stem
        return _safe_name(f"robomind_{embodiment}_{task_name}_{episode_id}")

    def _infer_embodiment(self, dataset_name: Optional[str], root: h5py.File) -> str:
        if dataset_name in self.schema_map:
            return str(dataset_name)

        rgb_group = root["observations"]["rgb_images"]
        camera_names = sorted(rgb_group.keys())

        if root.attrs.get("sim", False) and "franka" in root:
            return "h5_simulation"
        if camera_names == ["camera_front", "camera_left_wrist", "camera_right_wrist"]:
            return "h5_agilex_3rgb"
        if camera_names == ["camera_left", "camera_right", "camera_top"]:
            return "h5_franka_3rgb"
        if camera_names == ["camera_top"]:
            master_group = root.get("master")
            puppet_group = root.get("puppet")
            if master_group is not None and "end_effector" in master_group:
                ee_width = master_group["end_effector"].shape[1]
                if ee_width == 12:
                    return "h5_tienkung_xsens_1rgb"
            if master_group is not None and "joint_position" in master_group:
                width = master_group["joint_position"].shape[1]
                if width == 16:
                    return "h5_tienkung_gello_1rgb"
                if width == 7:
                    return "h5_ur_1rgb"
                if width == 8 and puppet_group is not None and "end_effector" in puppet_group:
                    return "h5_franka_1rgb"
        raise ValueError(
            f"Unable to infer RoboMIND embodiment from dataset_name={dataset_name}, "
            f"camera_names={camera_names}, root_keys={list(root.keys())}"
        )

    def _task_language(self, meta: Dict[str, Optional[str]], root: h5py.File) -> str:
        # Prefer the natural-language instruction stored in the HDF5
        if "language_raw" in root:
            lang = root["language_raw"][0]
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            lang = lang.strip()
            if lang:
                return lang
        base = meta["task_name"] or meta["dataset_name"] or "robomind_task"
        return base.replace("_h5", "").replace("_", " ").strip()

    def _lookup_intrinsics(
        self, embodiment: str, cam_id: str, frame: np.ndarray
    ) -> Tuple[Dict[str, float], str]:
        embodiment_intrinsics = self.intrinsics_map.get(embodiment, {})
        for candidate in _intrinsics_camera_aliases(cam_id):
            if candidate in embodiment_intrinsics:
                return embodiment_intrinsics[candidate], f"RoboMIND_intrinsics:{embodiment}"

        h, w = frame.shape[:2]
        return {
            "fx": float(w),
            "fy": float(h),
            "cx": float(w / 2.0),
            "cy": float(h / 2.0),
        }, "image_shape_fallback"

    @staticmethod
    def _lookup_extrinsics() -> Tuple[np.ndarray, str]:
        """RoboMIND publishes no extrinsics: identity."""
        return np.eye(4, dtype=np.float64), "identity_debug_unavailable"

    def _extract_master_gripper(self, root: h5py.File, embodiment: str) -> np.ndarray:
        if embodiment in {"h5_franka_1rgb", "h5_franka_3rgb", "h5_ur_1rgb"}:
            return np.asarray(root["master"]["joint_position"][:, -1], dtype=np.float64)

        if embodiment == "h5_agilex_3rgb":
            left = np.asarray(root["master"]["joint_position_left"][:, -1], dtype=np.float64)
            right = np.asarray(root["master"]["joint_position_right"][:, -1], dtype=np.float64)
            return np.maximum(left, right)

        if embodiment == "h5_tienkung_gello_1rgb":
            joint_position = np.asarray(root["master"]["joint_position"][:], dtype=np.float64)
            return np.maximum(joint_position[:, 7], joint_position[:, 15])

        if embodiment == "h5_tienkung_xsens_1rgb":
            end_effector = np.asarray(root["master"]["end_effector"][:], dtype=np.float64)
            return np.mean(np.abs(end_effector), axis=1)

        if embodiment == "h5_simulation":
            return np.asarray(root["franka"]["joint_position"][:, -1], dtype=np.float64)

        raise NotImplementedError(f"Unsupported RoboMIND gripper extraction for {embodiment}")

    @staticmethod
    def _select_active_arm(root: h5py.File) -> str:
        """Bimanual AgileX: the arm whose end effector moves more."""
        left_ee = np.asarray(root["puppet"]["end_effector_left"][:, :3], dtype=np.float64)
        right_ee = np.asarray(root["puppet"]["end_effector_right"][:, :3], dtype=np.float64)
        left_motion = float(np.linalg.norm(left_ee[-1] - left_ee[0]))
        right_motion = float(np.linalg.norm(right_ee[-1] - right_ee[0]))
        return "left" if left_motion >= right_motion else "right"

    def _extract_ee_positions(self, root: h5py.File, embodiment: str) -> np.ndarray:
        """End-effector xyz per state frame (N, 3). Used as both tip and flange."""
        if embodiment in {"h5_franka_1rgb", "h5_franka_3rgb", "h5_ur_1rgb"}:
            # puppet/end_effector: (N, 6) = (x,y,z,rx,ry,rz) euler-xyz
            ee = np.asarray(root["puppet"]["end_effector"][:], dtype=np.float64)
            return ee[:, :3]

        if embodiment == "h5_agilex_3rgb":
            arm = self._select_active_arm(root)
            # puppet/end_effector_{arm}: (N, 7) -- position + orientation
            ee = np.asarray(root["puppet"][f"end_effector_{arm}"][:], dtype=np.float64)
            return ee[:, :3]

        if embodiment == "h5_simulation":
            ee = np.asarray(root["franka"]["end_effector"][:], dtype=np.float64)
            return ee[:, :3]

        if embodiment in {"h5_tienkung_gello_1rgb", "h5_tienkung_xsens_1rgb"}:
            # Tien Kung only has joint angles, no Cartesian EE data: zeros.
            n = root["observations"]["rgb_images"][
                list(root["observations"]["rgb_images"].keys())[0]
            ].shape[0]
            return np.zeros((n, 3), dtype=np.float64)

        raise NotImplementedError(
            f"RoboMIND embodiment {embodiment} does not expose a supported Cartesian "
            "end-effector trajectory."
        )

    def load_episode(self, episode_path: str) -> EpisodeData:
        trajectory_path = Path(episode_path)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"RoboMIND trajectory not found: {trajectory_path}")

        root = h5py.File(trajectory_path, "r")
        meta = _parse_metadata(self.root, trajectory_path)
        embodiment = self._infer_embodiment(meta["dataset_name"], root)

        rgb_group = root["observations"]["rgb_images"]
        all_camera_names = sorted(rgb_group.keys())
        # Filter out arm-mounted (moving) cameras -- pipeline requires static cameras
        moving = MOVING_CAMERAS.get(embodiment, set())
        camera_names = [c for c in all_camera_names if c not in moving]
        if not camera_names:
            root.close()
            raise ValueError(
                f"No static cameras remain after filtering for {embodiment} "
                f"(all cameras: {all_camera_names}, moving: {moving})"
            )
        camera_lengths = [len(rgb_group[cam]) for cam in camera_names]
        num_frames = int(min(camera_lengths))
        if num_frames <= 0:
            root.close()
            raise ValueError(f"No RGB frames found in RoboMIND episode: {trajectory_path}")

        # The master gripper signal is not used for intervals; it only bounds
        # the usable state length.
        master_gripper_state = self._extract_master_gripper(root, embodiment)
        ee_positions_state = self._extract_ee_positions(root, embodiment)

        state_len = min(len(master_gripper_state), ee_positions_state.shape[0])
        if state_len <= 0:
            root.close()
            raise ValueError(f"No usable robot state found in RoboMIND episode: {trajectory_path}")

        if state_len == num_frames:
            state_frame_indices = np.arange(num_frames, dtype=np.int64)
        else:
            state_frame_indices = np.linspace(
                0, state_len - 1, num=num_frames
            ).round().astype(np.int64)

        ee_positions_frame = ee_positions_state[state_frame_indices]

        camera_infos: List[CameraInfo] = []
        for cam_id in camera_names:
            sample_frame = _decode_frame(rgb_group[cam_id][0], embodiment, "rgb_images")
            intrinsics, intrinsic_source = self._lookup_intrinsics(embodiment, cam_id, sample_frame)
            T_base_to_cam, extrinsic_source = self._lookup_extrinsics()
            camera_infos.append(
                CameraInfo(
                    cam_id=cam_id,
                    intrinsics=intrinsics,
                    T_base_to_cam=T_base_to_cam,
                    extrinsic_source=extrinsic_source,
                    intrinsic_source=intrinsic_source,
                    distortion=None,
                )
            )

        # Build annotation ID for step-annotation lookup:
        # matches JSON "id" field, e.g. h5_agilex_3rgb/task/success_episodes/train/ep_id/data
        ann_id = None
        try:
            ann_id = str(trajectory_path.parent.relative_to(self.root))
        except ValueError:
            pass

        return EpisodeData(
            num_frames=num_frames,
            fps=self.default_fps,
            camera_infos=camera_infos,
            extra={
                "h5": root,
                "embodiment": embodiment,
                "task_language": self._task_language(meta, root),
                "ee_positions_frame": np.asarray(ee_positions_frame, dtype=np.float64),
                "annotation_id": ann_id,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        # Use per-step language annotations if available
        ann_id = episode_data.extra.get("annotation_id")
        ann = self.step_annotations.get(ann_id) if ann_id else None

        if ann and ann.get("steps"):
            intervals = []
            steps = ann["steps"]
            for i, step in enumerate(steps):
                start = self._parse_frame_number(step.get("start_frame", ""))
                end = self._parse_frame_number(step.get("end_frame", ""))
                # Clamp to valid range
                start = max(0, min(start, episode_data.num_frames - 1))
                end = max(start, min(end, episode_data.num_frames - 1))

                # Skip single-frame steps (too short for trajectory)
                min_frames = max(2, int(self.min_interval_time * episode_data.fps))
                if end - start < min_frames:
                    continue

                iv = Interval(
                    start=start,
                    end=end,
                    language=step.get("step_description", ""),
                    contact_frame=start,
                )
                # obs_frame: use the frame before the step starts
                if not intervals:
                    iv.obs_frame = 0
                else:
                    iv.obs_frame = max(0, min(intervals[-1].end + 1, start))
                intervals.append(iv)

            if intervals:
                return intervals
            # Fall through if annotation parsing produced nothing

        # Fallback: whole episode as single interval
        intervals = [
            Interval(
                start=0,
                end=max(0, episode_data.num_frames - 1),
                language=episode_data.extra["task_language"],
                obs_frame=0,
                contact_frame=0,
            )
        ]
        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        root: h5py.File = episode_data.extra["h5"]
        embodiment = episode_data.extra["embodiment"]

        rgb_dataset = root["observations"]["rgb_images"][cam_id]
        rgb = _decode_frame(rgb_dataset[frame_idx], embodiment, "rgb_images")

        depth = None
        depth_group = root["observations"].get("depth_images")
        if self._extract_depth and depth_group is not None and cam_id in depth_group:
            depth_dataset = depth_group[cam_id]
            if frame_idx < len(depth_dataset):
                decoded_depth = _decode_frame(
                    depth_dataset[frame_idx], embodiment, "depth_images"
                )
                depth = np.asarray(decoded_depth, dtype=np.float32)
                if depth.size > 0 and np.nanmax(depth) <= 100:
                    depth = depth * 1000.0
        return rgb, depth

    def get_gripper_tip_3d(self, episode_data: EpisodeData, frame_idx: int) -> np.ndarray:
        """End-effector position (no tip offset)."""
        return episode_data.extra["ee_positions_frame"][frame_idx].astype(np.float64)

    def get_flange_3d(self, episode_data: EpisodeData, frame_idx: int) -> np.ndarray:
        return episode_data.extra["ee_positions_frame"][frame_idx].astype(np.float64)

    def compute_trajectory(
        self, episode_data: EpisodeData, interval: Interval, cam_info: CameraInfo
    ) -> List[TrajectoryPoint]:
        traj = super().compute_trajectory(episode_data, interval, cam_info)
        # With identity extrinsics the base-frame points are projected as if
        # they were in the camera frame, so the pixels can land outside the
        # image; rescale them into the image (affects trajectory_2d/motion_2d).
        if cam_info.extrinsic_source == "identity_debug_unavailable":
            traj = self._rescale_pixels_to_image(traj, episode_data, cam_info)
        return traj

    @staticmethod
    def _rescale_pixels_to_image(
        traj: List[TrajectoryPoint],
        episode_data: EpisodeData,
        cam_info: CameraInfo,
    ) -> List[TrajectoryPoint]:
        """Rescale pixel projections into the image when any falls outside it."""
        pixels = [(p.position_pixel[0], p.position_pixel[1])
                  for p in traj if p.position_pixel is not None]
        if len(pixels) < 2:
            return traj

        xs, ys = zip(*pixels)
        img_w = cam_info.intrinsics["cx"] * 2
        img_h = cam_info.intrinsics["cy"] * 2
        margin = 0.1  # 10% border

        # Check if all points are already within the image
        if (min(xs) >= 0 and max(xs) < img_w and
                min(ys) >= 0 and max(ys) < img_h):
            return traj

        # Rescale to fit within image with margin
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_span = max(x_max - x_min, 1e-6)
        y_span = max(y_max - y_min, 1e-6)
        usable_w = img_w * (1 - 2 * margin)
        usable_h = img_h * (1 - 2 * margin)

        for p in traj:
            if p.position_pixel is not None:
                nx = (p.position_pixel[0] - x_min) / x_span * usable_w + img_w * margin
                ny = (p.position_pixel[1] - y_min) / y_span * usable_h + img_h * margin
                p.position_pixel = [int(round(nx)), int(round(ny))]
        return traj

    def close_episode(self, episode_data: EpisodeData) -> None:
        root = episode_data.extra.get("h5")
        if root is not None:
            root.close()
