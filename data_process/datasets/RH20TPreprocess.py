"""
RH20T preprocessing, robot mode and human-only mode (``human_only``).

Each scene folder (``{root}/RH20T_cfgN/task_XXXX_user_..._scene_..._cfg_XXXX``)
is read through the RH20T API (datasets/extern/rh20t_api), which provides
per-camera timestamps, intrinsics, base-aligned extrinsics, camera-aligned TCP
poses and the gripper command. Scenes are selected by matching the English task
names in ``task_txt_path`` against task_description.json; ``human`` scenes are
kept only in human-only mode. If a scene only has color.mp4, frames are first
extracted in place with the RH20T API.

Robot mode: intervals run from a gripper close to the next open in the
smoothed gripper command, extended by 10 frames after release; intervals that
are too short or where the tip moves < 2 cm (3D) or < 15 px (2D) are dropped.
In-hand cameras are excluded. Human-only mode: one interval per scene covering
the whole episode, no trajectory.

FPS is fixed at 10. Intrinsics are rescaled to image_width x image_height for
projection. Depth PNGs are returned as-is (float32).
"""

import json
import re
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import CameraInfo, EpisodeData, Interval, TrajectoryPoint

from .extern.rh20t_api.rh20t_api.configurations import load_conf
from .extern.rh20t_api.rh20t_api.scene import RH20TScene
from .extern.rh20t_api.rh20t_api.extract import convert_scene


# ================================================================== #
# RH20TPreprocess
# ================================================================== #

class RH20TPreprocess(RoboDatasetBase):
    """RH20T dataset preprocessor.

    Loads each scene via the RH20T Scene API, which provides per-camera
    timestamps, intrinsics, base-aligned extrinsics, TCP poses, and a
    gripper-command signal. Manipulation intervals are detected from
    close/open transitions in the smoothed gripper-command signal.

    Two modes (selected by ``cfg.dataset.human_only``):
      * Robot mode: keeps only static (non in-hand) cameras and filters
        intervals by minimum length and gripper-tip motion.
      * Human-only mode: emits one full-episode interval per scene with
        no gripper-based filtering.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.dataset_cfg = dataset
        self.dataset_root = Path(dataset.root)
        raw_subset = dataset.subset
        if isinstance(raw_subset, str):
            self.subsets = [raw_subset]
        else:
            self.subsets = list(raw_subset)
        self.subset_paths: List[Path] = []
        for s in self.subsets:
            p = self.dataset_root / s
            if not p.exists():
                raise FileNotFoundError(f"Subset {s} not found in {self.dataset_root}")
            self.subset_paths.append(p)
        self.task_desc_path = self.dataset_root / "task_description.json"
        with self.task_desc_path.open("r", encoding="utf-8") as f:
            self.task_desc = json.load(f)
        self._extract_depth = bool(getattr(dataset, "extract_depth", True))
        self.human_only = bool(getattr(dataset, "human_only", False))

    # ------------------------------------------------------------------ #
    # Abstract implementations
    # ------------------------------------------------------------------ #

    def get_task_id_from_tasks(self, task_txt_path: str) -> List[str]:
        """Map English task descriptions in txt to RH20T task IDs."""
        def _normalize_task_text(text: str) -> str:
            return " ".join(text.strip().lower().split())

        task_txt = Path(task_txt_path)
        with task_txt.open("r", encoding="utf-8") as f:
            task_strings = [line.strip() for line in f if line.strip()]

        if not task_strings:
            raise ValueError(f"No tasks found in task list file: {task_txt_path}")

        english_to_task_ids: Dict[str, List[str]] = {}
        for task_id, task_info in self.task_desc.items():
            if not isinstance(task_info, dict):
                continue
            task_english = task_info.get("task_description_english")
            if not task_english or task_english == "?":
                continue
            normalized = _normalize_task_text(task_english)
            english_to_task_ids.setdefault(normalized, []).append(task_id)

        task_ids: List[str] = []
        missing_tasks: List[str] = []
        for task_text in task_strings:
            matched_ids = english_to_task_ids.get(_normalize_task_text(task_text))
            if not matched_ids:
                missing_tasks.append(task_text)
                continue

            for task_id in matched_ids:
                if task_id not in task_ids:
                    task_ids.append(task_id)

        if missing_tasks:
            missing = ", ".join(missing_tasks)
            print(
                f"[RH20T] Task(s) not found in task_description.json (English): {missing}"
            )

        if not task_ids:
            raise ValueError("[RH20T] No valid task IDs were resolved from task list.")

        return task_ids

    def discover_episodes(self) -> List[str]:
        """Return scene paths under the configured subset(s).

        Episodes are filtered by the English task list at
        ``cfg.dataset.task_txt_path`` and by ``human_only`` (which
        either keeps or excludes the human-perspective scenes).
        """
        print(f"[RH20T] Discovering episodes from {self.cfg.dataset.task_txt_path}")
        task_ids = self.get_task_id_from_tasks(self.cfg.dataset.task_txt_path)
        task_id_pattern = re.compile("|".join(re.escape(task_id) for task_id in task_ids))

        if self.human_only:
            all_episodes_paths = sorted(
                str(p)
                for subset_path in self.subset_paths
                for p in subset_path.iterdir()
                if p.is_dir() and task_id_pattern.search(p.name) and "human" in p.name
            )
        else:
            all_episodes_paths = sorted(
                str(p)
                for subset_path in self.subset_paths
                for p in subset_path.iterdir()
                if p.is_dir() and task_id_pattern.search(p.name) and "human" not in p.name
            )

        print(f"[RH20T] Found {len(all_episodes_paths)} episodes")
        return all_episodes_paths

    def get_episode_id(self, episode_path: str) -> str:
        """Return ``RH20T_<scene_folder_name>`` as the episode id."""
        base_name = Path(episode_path).name
        return f"RH20T_{base_name}"

    @staticmethod
    def _to_intrinsics_dict(K: np.ndarray) -> Dict[str, float]:
        K = np.asarray(K, dtype=np.float64)
        if K.ndim != 2 or K.shape[0] < 2 or K.shape[1] < 3:
            raise ValueError(f"Invalid intrinsic matrix shape: {K.shape}")
        return {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }

    def _ensure_images_extracted(self, scene_path: Path) -> None:
        """Convert mp4 to images if any camera is missing extracted frames."""
        cam_dirs = sorted(
            p for p in scene_path.iterdir() if p.is_dir() and p.name.startswith("cam_")
        )
        if not cam_dirs:
            return

        all_extracted = all(
            (cam / "color").exists() and any((cam / "color").glob("*.jpg"))
            for cam in cam_dirs
        )
        if all_extracted:
            return

        # Check that source mp4 files exist for conversion.
        has_source = any(
            (cam / "color.mp4").exists() and (cam / "timestamps.npy").exists()
            for cam in cam_dirs
        )
        if not has_source:
            raise FileNotFoundError(
                f"No extracted images and no color.mp4 source found in: {scene_path}"
            )

        print(f"[RH20T] Images not extracted for {scene_path}, converting from mp4...")

        # Infer extraction size from the first available color.mp4.
        probe_path = next(cam / "color.mp4" for cam in cam_dirs if (cam / "color.mp4").exists())
        cap = cv2.VideoCapture(str(probe_path))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Failed to infer frame size from {probe_path}")
        extract_size = (width, height)

        # Use co-located depth.mp4 if any camera has one.
        has_depth = any((cam / "depth.mp4").exists() for cam in cam_dirs)
        depth_dir = str(scene_path) if has_depth else None

        convert_scene(
            scene_root_dir=str(scene_path),
            dest_scene_dir=str(scene_path),
            scene_depth_dir=depth_dir,
            size=extract_size,
        )

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Open the RH20T scene and assemble per-camera streams.

        Builds a ``CameraInfo`` for each camera that has valid
        timestamps, intrinsics, and base-aligned extrinsics, trims them
        to a common frame count, and pre-loads the gripper-command
        signal (robot mode only) for later interval detection.
        """
        ep = Path(episode_path)
        if not ep.exists():
            raise FileNotFoundError(f"Episode folder not found: {episode_path}")

        self._ensure_images_extracted(ep)

        configs_path = (
            Path(__file__).resolve().parent / "extern" / "rh20t_api" / "configs" / "configs.json"
        )
        if not configs_path.exists():
            raise FileNotFoundError(f"RH20T config file not found: {configs_path}")

        robot_configs = load_conf(str(configs_path))
        scene = RH20TScene(str(ep), robot_configs)

        all_serials = list(scene.serials)
        if not all_serials:
            raise ValueError(f"No camera serials found in scene: {episode_path}")

        intrinsics_by_serial = scene.intrinsics
        extrinsics_by_serial = scene.extrinsics_base_aligned
        raw_cam_timestamps: Dict[str, List[int]] = {
            serial: list(scene.low_freq_timestamps.get(serial, [])) for serial in all_serials
        }

        valid_serials = [
            serial
            for serial in all_serials
            if raw_cam_timestamps.get(serial)
            and serial in intrinsics_by_serial
            and serial in extrinsics_by_serial
        ]
        if not valid_serials:
            raise ValueError(
                f"No valid camera stream found (timestamps + intrinsics + extrinsics): {episode_path}"
            )

        # Keep index space shared across all cameras by trimming to common length.
        num_frames = min(len(raw_cam_timestamps[serial]) for serial in valid_serials)
        if num_frames <= 0:
            raise ValueError(f"Invalid frame count (0) for scene: {episode_path}")

        camera_timestamps: Dict[str, List[int]] = {
            serial: raw_cam_timestamps[serial][:num_frames] for serial in valid_serials
        }

        camera_infos: List[CameraInfo] = []
        for serial in valid_serials:
            K = intrinsics_by_serial[serial]
            T_base_to_cam = np.asarray(extrinsics_by_serial[serial], dtype=np.float64)
            if T_base_to_cam.shape != (4, 4):
                raise ValueError(
                    f"Invalid extrinsic shape for serial {serial}: {T_base_to_cam.shape}"
                )
            camera_infos.append(
                CameraInfo(
                    cam_id=serial,
                    intrinsics=self._to_intrinsics_dict(K),
                    T_base_to_cam=T_base_to_cam,
                    extrinsic_source="scene.extrinsics_base_aligned",
                    intrinsic_source="scene.intrinsics",
                    distortion=None,
                )
            )

        fps = 10.0
        # Pre-load gripper command trace (robot mode only).
        if self.human_only:
            all_gripper_ts = []
            gripper_command = []
        else:
            all_gripper_ts = sorted(
                {int(ts) for serial_gripper in scene.gripper.values() for ts in serial_gripper.keys()}
            )
            gripper_command: List[float] = []
            for ts in all_gripper_ts:
                try:
                    gripper_command.append(float(scene.get_gripper_command(ts)))
                except Exception:
                    gripper_command.append(float("nan"))

        task_id = Path(episode_path).name.split("_")[1]

        task_name = self.task_desc[f"task_{task_id}"]["task_description_english"]

        return EpisodeData(
            num_frames=num_frames,
            fps=fps,
            camera_infos=camera_infos,
            extra={
                "scene": scene,
                "camera_timestamps": camera_timestamps,
                "gripper_timestamps": all_gripper_ts,
                "gripper_command": np.asarray(gripper_command, dtype=np.float64),
                "task_name": task_name,
            },
        )


    def scale_intrinsics_for_image(self, K_raw: Dict[str, Any], w: int, h: int) -> Dict[str, Any]:
        """Rescale K to a w x h image, assuming the principal point is the calibration image center."""
        fx, fy = float(K_raw['fx']), float(K_raw['fy'])
        cx, cy = float(K_raw['cx']), float(K_raw['cy'])
        calib_w = 2.0 * cx if cx > 0 else float(w)
        calib_h = 2.0 * cy if cy > 0 else float(h)
        sx = float(w) / calib_w
        sy = float(h) / calib_h
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float64)
        return self._to_intrinsics_dict(K)

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """One interval per gripper close->open pair, with all static cameras.

        Gripper timestamps are mapped to frames through the first static camera.
        """
        camera_timestamps: Dict[str, List[int]] = episode_data.extra.get("camera_timestamps", {})
        if not camera_timestamps:
            return []

        if self.human_only:
            # Single interval: entire episode, obs_frame = first frame.
            all_cam_ids = [
                cid for cid in camera_timestamps
                if len(camera_timestamps[cid]) > 0
            ]
            if not all_cam_ids:
                return []
            all_cams = [c for c in episode_data.camera_infos if c.cam_id in all_cam_ids]
            if not all_cams:
                return []
            return [
                Interval(
                    start=0,
                    end=episode_data.num_frames - 1,
                    obs_frame=0,
                    contact_frame=0,
                    language=episode_data.extra["task_name"],
                    camera_infos=all_cams,
                )
            ]

        # Collect static (non-in-hand) cameras.
        in_hand = set(episode_data.extra["scene"].in_hand_serials)
        static_cam_ids = [
            cid for cid in camera_timestamps
            if cid not in in_hand and len(camera_timestamps[cid]) > 0
        ]
        if not static_cam_ids:
            return []
        static_cams = [c for c in episode_data.camera_infos if c.cam_id in static_cam_ids]
        if not static_cams:
            return []

        # Use the first static camera as reference for timestamp->frame mapping.
        ref_cam_id = static_cams[0].cam_id
        ref_ts = np.asarray(camera_timestamps[ref_cam_id], dtype=np.int64)

        gripper_ts = np.asarray(episode_data.extra.get("gripper_timestamps", []), dtype=np.int64)
        gripper_cmd = np.asarray(episode_data.extra.get("gripper_command", []), dtype=np.float64)

        min_interval_frames = max(
            1,
            int(float(getattr(self.cfg.dataset, "min_interval_time", 0.0)) * episode_data.fps),
        )

        # --- detect all (close_ts, open_ts) pairs from the gripper signal ---
        close_open_pairs: List[Tuple[int, int]] = []

        if (
            gripper_ts.size > 0
            and gripper_cmd.size == gripper_ts.size
            and np.any(np.isfinite(gripper_cmd))
        ):
            cmd = gripper_cmd.copy()
            if cmd.size >= 5:
                kernel = np.ones(5, dtype=np.float64) / 5.0
                cmd = np.convolve(cmd, kernel, mode="same")

            cmd_min = float(np.nanmin(cmd))
            cmd_max = float(np.nanmax(cmd))
            close_ratio = float(getattr(self.cfg.dataset, "gripper_threshold", 0.2))
            close_ratio = float(np.clip(close_ratio, 0.0, 1.0))
            close_threshold_val = cmd_min + close_ratio * (cmd_max - cmd_min)

            is_closed = cmd <= close_threshold_val

            if cmd.size > 1:
                close_crossings = np.where((~is_closed[:-1]) & (is_closed[1:]))[0] + 1
                open_crossings = np.where((is_closed[:-1]) & (~is_closed[1:]))[0] + 1

                if close_crossings.size > 0:
                    used_up_to = -1
                    for ci_idx in close_crossings:
                        if ci_idx <= used_up_to:
                            continue
                        later_opens = open_crossings[open_crossings > ci_idx]
                        if later_opens.size > 0:
                            oi_idx = int(later_opens[0])
                            used_up_to = oi_idx
                        else:
                            oi_idx = len(cmd) - 1  # end of signal
                        close_open_pairs.append((int(gripper_ts[ci_idx]), int(gripper_ts[oi_idx])))
                else:
                    # Fallback: strongest single close event -> end of signal.
                    ci_idx = int(np.argmin(np.diff(cmd)) + 1)
                    close_open_pairs.append((int(gripper_ts[ci_idx]), int(gripper_ts[-1])))

        if not close_open_pairs:
            # No usable gripper signal -- treat the whole episode as one interval.
            close_open_pairs.append((int(ref_ts[0]), int(ref_ts[-1])))

        # --- map each pair to frame indices and build Intervals -----------
        num_frames = episode_data.num_frames
        intervals: List[Interval] = []
        for idx_in_pairs, (close_ts, open_ts) in enumerate(close_open_pairs):
            start = int(np.argmin(np.abs(ref_ts - close_ts)))
            end = int(np.argmin(np.abs(ref_ts - open_ts)))
            end = max(end, start)

            # Extend end by 10 frames to capture post-release motion,
            # but clamp before the next close to avoid overlapping intervals.
            if idx_in_pairs + 1 < len(close_open_pairs):
                next_start = int(np.argmin(np.abs(ref_ts - close_open_pairs[idx_in_pairs + 1][0])))
                end = min(end + 10, next_start - 1, num_frames - 1)
            else:
                end = min(end + 10, num_frames - 1)

            if end - start + 1 < min_interval_frames:
                continue

            # Skip intervals where the gripper barely moves: compare 3D
            # gripper-tip positions at start vs end and their 2D projections.
            try:
                tip_start = self.get_gripper_tip_3d(episode_data, start, ref_cam_id)
                tip_end = self.get_gripper_tip_3d(episode_data, end, ref_cam_id)
                dist_3d = float(np.linalg.norm(tip_end - tip_start))
                # Also check 2D projection distance via the reference camera.
                ref_cam = static_cams[0]
                w, h = self.dataset_cfg.image_width, self.dataset_cfg.image_height
                scaled_K = self.scale_intrinsics_for_image(ref_cam.intrinsics, w=w, h=h)
                px_start = self.project_to_image(tip_start, scaled_K)
                px_end = self.project_to_image(tip_end, scaled_K)
                dist_2d = (float(np.linalg.norm(
                    np.array(px_end, dtype=np.float32) - np.array(px_start, dtype=np.float32)
                )) if px_start is not None and px_end is not None else 0.0)
                if dist_3d < 0.02 or dist_2d < 15:
                    continue
            except Exception:
                pass  # keep the interval if we can't compute distances

            contact_frame = start
            obs_frame = 0 if not intervals else intervals[-1].end + 1

            intervals.append(
                Interval(
                    start=start,
                    end=end,
                    obs_frame=obs_frame,
                    contact_frame=contact_frame,
                    language=episode_data.extra["task_name"],
                    camera_infos=static_cams,
                )
            )

        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return RGB + optional depth at an indexed low-frequency frame."""
        scene: RH20TScene = episode_data.extra["scene"]
        camera_timestamps: Dict[str, List[int]] = episode_data.extra.get("camera_timestamps")

        if cam_id not in camera_timestamps:
            raise KeyError(f"Camera '{cam_id}' not found in camera_timestamps.")

        cam_ts = camera_timestamps[cam_id]
        if frame_idx < 0 or frame_idx >= len(cam_ts):
            raise IndexError(
                f"frame_idx {frame_idx} out of range for camera '{cam_id}' "
                f"(num_frames={len(cam_ts)})."
            )
        timestamp = int(cam_ts[frame_idx])

        image_types = ["color", "depth"] if self._extract_depth else ["color"]
        pair = scene.get_image_path_pairs(timestamp, image_types=image_types).get(cam_id, [])
        if not pair:
            raise FileNotFoundError(
                f"No image pair found for camera '{cam_id}' at timestamp {timestamp}."
            )

        color_path = Path(pair[0])
        if not color_path.exists():
            fallback_color = sorted(color_path.parent.glob("*.jpg"))
            if not fallback_color:
                raise FileNotFoundError(f"No color jpg found in: {color_path.parent}")
            color_path = fallback_color[0]

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Failed to read color image: {color_path}")
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        depth: Optional[np.ndarray] = None
        if self._extract_depth:
            depth_path = (
                Path(pair[1])
                if len(pair) > 1
                else (color_path.parent.parent / "depth" / f"{timestamp}.png")
            )

            if not depth_path.exists():
                fallback_depth = sorted(depth_path.parent.glob("*.png"))
                if fallback_depth:
                    depth_path = fallback_depth[0]
                else:
                    return rgb, None

            depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is not None:
                if depth_raw.ndim == 3:
                    depth_raw = depth_raw[..., 0]
                depth = depth_raw.astype(np.float32)

        return rgb, depth

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int, cam_id: str
    ) -> np.ndarray:
        """Return the TCP position [x, y, z] in camera ``cam_id``'s frame (meters).

        Unlike the base-class contract this is already camera-aligned
        (``scene.get_tcp_aligned``), so compute_trajectory applies no extrinsic.
        """
        scene: RH20TScene = episode_data.extra["scene"]
        camera_timestamps: Dict[str, List[int]] = episode_data.extra.get("camera_timestamps")

        ts = camera_timestamps[cam_id]
        if frame_idx < 0 or frame_idx >= len(ts):
            raise IndexError(
                f"frame_idx {frame_idx} out of range (num_frames={len(ts)})."
            )
        timestamp = int(ts[frame_idx])

        tcp_cam = np.asarray(scene.get_tcp_aligned(timestamp, serial=cam_id), dtype=np.float64).reshape(-1)
        if tcp_cam.size < 3:
            raise ValueError(f"Invalid TCP shape from get_tcp_aligned: {tcp_cam.shape}")

        return tcp_cam[:3].copy()

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int, cam_id: str = None
    ) -> np.ndarray:
        # RH20T tip == flange (no offset applied)
        return self.get_gripper_tip_3d(episode_data, frame_idx, cam_id)

    def compute_contact_flange_point(
        self, episode_data: EpisodeData, contact_frame_idx: int, cam: CameraInfo,
    ) -> TrajectoryPoint:
        """Project the contact-frame flange position into the given camera."""
        flange_cam = self.get_flange_3d(episode_data, contact_frame_idx, cam.cam_id)
        w, h = self.dataset_cfg.image_width, self.dataset_cfg.image_height
        pixel = self.project_to_image(flange_cam, self.scale_intrinsics_for_image(cam.intrinsics, w=w, h=h))
        return TrajectoryPoint(
            frame_idx=contact_frame_idx,
            position_3d=flange_cam.tolist(),
            position_pixel=pixel,
        )

    def compute_trajectory(
        self, episode_data: EpisodeData, interval: Interval,
        cam_info: CameraInfo,
    ) -> List[TrajectoryPoint]:
        """TCP trajectory in the camera frame, projected with rescaled intrinsics."""
        trajectory: List[TrajectoryPoint] = []
        for idx in range(interval.start, interval.end + 1):
            tip_cam = self.get_gripper_tip_3d(episode_data, idx, cam_info.cam_id)
            w, h = self.dataset_cfg.image_width, self.dataset_cfg.image_height
            pixel = self.project_to_image(tip_cam, self.scale_intrinsics_for_image(cam_info.intrinsics, w=w, h=h))
            trajectory.append(TrajectoryPoint(frame_idx=idx, position_3d=tip_cam.tolist(), position_pixel=pixel))
        return trajectory

    def _process_episode(self, episode_path: str) -> Optional[Dict]:
        """Process one episode. In human_only mode, skip trajectory/motion/flange."""
        if not self.human_only:
            return super()._process_episode(episode_path)

        print(f"\n{'=' * 60}\n  [HUMAN] {episode_path}\n{'=' * 60}")

        episode_data = self.load_episode(episode_path)
        episode_id = self.get_episode_id(episode_path)
        intervals = self.detect_intervals(episode_data)

        if not intervals:
            print("  No intervals found, skipping.")
            self.close_episode(episode_data)
            return None

        fps = episode_data.fps
        episode_dir = self.output_root / episode_id
        print(
            f"num of frames={episode_data.num_frames}  num of cams={len(episode_data.camera_infos)}  "
            f"num of intervals={len(intervals)}  fps={fps}"
        )

        for i, interval in enumerate(intervals):
            interval_dir = episode_dir / f"interval_{i:03d}"
            cam_meta: Dict[str, Any] = {}

            cams = interval.camera_infos if interval.camera_infos is not None else episode_data.camera_infos
            for ci, cam in enumerate(cams):
                cam_dir = interval_dir / f"cam_{ci}"
                cam_dir.mkdir(parents=True, exist_ok=True)

                obs_frame_idx = interval.obs_frame if interval.obs_frame is not None else interval.start
                contact_frame_idx = interval.contact_frame if interval.contact_frame is not None else interval.start

                obs_frame, depth = self.extract_frame_and_depth(
                    episode_data, cam.cam_id, obs_frame_idx
                )
                contact_frame = obs_frame if contact_frame_idx == obs_frame_idx else \
                    self.extract_frame_and_depth(episode_data, cam.cam_id, contact_frame_idx)[0]

                # Save images (obs_frame_traj = obs_frame with no overlay)
                self._save_images(cam_dir, obs_frame, contact_frame, obs_frame.copy())
                depth_src = self._save_depth_npy(cam_dir, depth)

                # Save trajectory.json with camera_info but empty trajectory/motion
                w, h = self.dataset_cfg.image_width, self.dataset_cfg.image_height
                scaled_K = self.scale_intrinsics_for_image(cam.intrinsics, w=w, h=h)
                data = {
                    'camera_info': {
                        'cam_id': cam.cam_id,
                        'intrinsics': scaled_K,
                        'T_base_to_cam': cam.T_base_to_cam.tolist(),
                        'extrinsic_source': cam.extrinsic_source,
                        'intrinsic_source': cam.intrinsic_source,
                        'distortion': cam.distortion,
                    },
                    'trajectory_3d': [],
                    'trajectory_2d': [{'frame_idx': obs_frame_idx, 'position_pixel': [w // 2, h // 2]}],
                    'motion_3d': None,
                    'motion_2d': {'start_pixel': [w // 2, h // 2], 'end_pixel': None},
                }
                with open(cam_dir / "trajectory.json", 'w') as f:
                    json.dump(data, f, indent=2)

                self._collect_cam_info_for_interval_metadata(cam_meta, ci, cam, depth_src)

            self._save_interval_metadata(interval_dir, episode_id, i, interval, cams, cam_meta)

        self._save_episode_summary(episode_dir, episode_id, episode_path, episode_data, intervals, cams, fps)
        self.close_episode(episode_data)
        print(f"Done: {episode_id}  ({len(intervals)} intervals)")
        return {'episode_id': episode_id, 'num_intervals': len(intervals)}

    def close_episode(self, episode_data: EpisodeData) -> None:
        """No-op. RH20T scene objects do not require explicit release."""
        pass
