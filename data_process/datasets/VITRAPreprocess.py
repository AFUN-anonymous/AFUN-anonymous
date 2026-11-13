"""
VITRA-1M preprocessing (EPIC-KITCHENS and Ego4D subsets).

Each VITRA episode .npy (``{root}/{subset}/episodic_annotations/*.npy``, or
``{root}/{subset}/{subset}/...``) is one interval. It provides MANO hand joints
in world space, per-frame SLAM world-to-camera extrinsics, intrinsics, language,
and the source-video frame indices; RGB frames are decoded from the source
videos under ``video_root``. The "gripper tip" is the index fingertip
(``tip_joint``) and the flange is the wrist. obs_frame is the first frame.

Contact frames:
- EPIC (``epic_contact_index``, tools/build_vitra_epic_contact_index.py):
  episodes without a hand-object contact are skipped.
- Ego4D (``ego4d_contact_index``, tools/build_vitra_ego4d_contact_index.py):
  when set, only indexed episodes are kept; when null, contact_frame = obs_frame.

The camera moves, so trajectories are expressed in the obs_frame camera
(compute_trajectory, reproject_camera_point_to_obs_frame). No depth is produced
here; mainprocess computes DA3 video depth.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, CameraInfo, EpisodeData, TrajectoryPoint

logger = logging.getLogger(__name__)

# MANO joint index: index fingertip -- used as "gripper tip"
JOINT_INDEX_TIP = 8


def _intrinsics_from_matrix(K: np.ndarray) -> Dict[str, float]:
    """Extract {fx, fy, cx, cy} from a 3x3 intrinsic matrix."""
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


class VITRAPreprocess(RoboDatasetBase):
    """VITRA-1M dataset preprocessor (see module docstring)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        ds = cfg.dataset

        self.root = Path(ds.root)
        self.video_root = Path(ds.video_root)
        self.subsets = list(ds.subsets)
        self.fps = float(ds.get("fps", 30.0))
        self.min_interval_frames = int(ds.get("min_interval_frames", 5))
        self.tip_joint = int(ds.get("tip_joint", JOINT_INDEX_TIP))

        # Ego4D contact frame index
        contact_index_path = ds.get("ego4d_contact_index")
        self._ego4d_contact_index: Dict[str, dict] = {}
        if contact_index_path:
            p = Path(contact_index_path)
            if p.exists():
                with open(p) as f:
                    self._ego4d_contact_index = json.load(f)
                logger.info("Loaded %d Ego4D contact index entries", len(self._ego4d_contact_index))
            else:
                logger.warning("ego4d_contact_index not found: %s", p)

        epic_contact_index_path = ds.get("epic_contact_index")
        self._epic_contact_index: Dict[str, dict] = {}
        if epic_contact_index_path:
            p = Path(epic_contact_index_path)
            if p.exists():
                with open(p) as f:
                    self._epic_contact_index = json.load(f)
                non_null = sum(1 for v in self._epic_contact_index.values() if v)
                logger.info(
                    "Loaded %d EPIC contact index entries (%d non-null)",
                    len(self._epic_contact_index), non_null,
                )
            else:
                logger.warning("epic_contact_index not found: %s", p)

    # ------------------------------------------------------------------ #
    # Abstract implementations
    # ------------------------------------------------------------------ #

    def discover_episodes(self) -> List[str]:
        """Discover all .npy episode annotation files across configured subsets.

        When ``ego4d_contact_index`` is configured, episodes whose stem is not
        present in the index are filtered out here so that ``detect_intervals``
        does not need to reject them later.
        """
        episodes = []
        for subset in self.subsets:
            # Two layouts: {root}/{subset}/episodic_annotations/ and
            # {root}/{subset}/{subset}/episodic_annotations/
            annot_dir = self.root / subset / "episodic_annotations"
            if not annot_dir.is_dir():
                annot_dir = self.root / subset / subset / "episodic_annotations"
            if not annot_dir.is_dir():
                logger.warning("Annotation dir not found: %s", annot_dir)
                continue
            npy_files = sorted(annot_dir.glob("*.npy"))
            episodes.extend(str(f) for f in npy_files)
        total = len(episodes)
        logger.info("Discovered %d episodes across %d subsets", total, len(self.subsets))

        if self._ego4d_contact_index:
            index_keys = self._ego4d_contact_index
            episodes = [p for p in episodes if Path(p).stem in index_keys]
            logger.info(
                "Filtered by ego4d_contact_index: %d / %d episodes kept",
                len(episodes), total,
            )

        if self._epic_contact_index:
            # Epic episodes without a contact_state match are dropped at
            # discovery. Entries with value==None are also dropped.
            epic_keys = {k for k, v in self._epic_contact_index.items() if v}
            before = len(episodes)
            kept = []
            for p in episodes:
                stem = Path(p).stem
                if stem.lower().startswith("epic_"):
                    if stem in epic_keys:
                        kept.append(p)
                else:
                    kept.append(p)
            episodes = kept
            logger.info(
                "Filtered by epic_contact_index: %d / %d episodes kept (epic matched=%d)",
                len(episodes), before, len(epic_keys),
            )
        return episodes

    def get_episode_id(self, episode_path: str) -> str:
        """Episode ID = .npy filename without extension (already unique)."""
        return Path(episode_path).stem

    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load a VITRA episode from its .npy annotation file."""
        ep = np.load(episode_path, allow_pickle=True).item()

        num_frames = len(ep["video_decode_frame"])
        active_hand = ep.get("anno_type", "right")

        # Determine which hand data to use
        hand_data = ep[active_hand]
        other_hand = "left" if active_hand == "right" else "right"

        # Validate kept_frames
        kept = np.asarray(hand_data["kept_frames"], dtype=int)
        valid_ratio = kept.sum() / max(len(kept), 1)
        if valid_ratio < 0.3:
            logger.warning(
                "Episode %s: only %.0f%% valid frames for %s hand",
                Path(episode_path).stem, valid_ratio * 100, active_hand,
            )

        # Camera intrinsics from 3x3 matrix
        K = ep["intrinsics"]  # (3, 3)
        intrinsics = _intrinsics_from_matrix(K)

        # Per-frame world-to-camera extrinsics (T, 4, 4), same convention as T_base_to_cam
        per_frame_ext = ep["extrinsics"]

        # Frame 0's extrinsic is the one written to CameraInfo
        T_base_to_cam = per_frame_ext[0].copy()

        # Validate extrinsic
        R = T_base_to_cam[:3, :3]
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            logger.warning(
                "Episode %s: extrinsic det(R)=%.4f, expected ~1.0",
                Path(episode_path).stem, det,
            )

        cam_info = CameraInfo(
            cam_id="ego_cam",
            intrinsics=intrinsics,
            T_base_to_cam=T_base_to_cam,
            extrinsic_source="vitra_slam_per_frame",
            intrinsic_source="vitra_droidcalib" if "ego4d" in episode_path.lower() else "vitra_calibration",
        )

        # Extract language
        language = ""
        for side in [active_hand, other_hand]:
            text_list = ep.get("text", {}).get(side, [])
            if text_list:
                language = text_list[0][0]
                break
        # Prefer rephrased text if original is empty
        if not language:
            for side in [active_hand, other_hand]:
                rephrase_list = ep.get("text_rephrase", {}).get(side, [])
                if rephrase_list and rephrase_list[0][0]:
                    language = rephrase_list[0][0][0]  # first rephrase variant
                    break

        # Determine source subset for video path resolution
        ep_stem = Path(episode_path).stem
        subset = self._detect_subset(ep_stem)

        return EpisodeData(
            num_frames=num_frames,
            fps=self.fps,
            camera_infos=[cam_info],
            extra={
                "episode_path": episode_path,
                "video_name": ep["video_name"],
                "video_decode_frame": ep["video_decode_frame"],
                "per_frame_extrinsics": per_frame_ext,
                "hand_data": hand_data,
                "language": language,
                "subset": subset,
                "kept_frames": kept,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Each VITRA episode is one interval."""
        T = episode_data.num_frames
        if T < self.min_interval_frames:
            logger.info("Episode too short (%d frames), skipping", T)
            return []

        # Check hand validity
        kept = episode_data.extra["kept_frames"]
        if kept.sum() < self.min_interval_frames:
            logger.info("Too few valid hand frames (%d), skipping", kept.sum())
            return []

        obs_frame = 0
        contact_frame = obs_frame
        language = episode_data.extra.get("language", "")
        ep_stem = Path(episode_data.extra["episode_path"]).stem
        subset = episode_data.extra.get("subset", "unknown")

        # EPIC subset -- require a contact_state lookup match.
        if subset == "epic":
            epic_match = self._epic_contact_index.get(ep_stem)
            if not epic_match:
                logger.debug("No EPIC contact-state match for %s, skipping", ep_stem)
                return []
            contact_frame = min(int(epic_match["contact_frame_local"]), T - 1)
            return [Interval(
                start=0,
                end=T - 1,
                obs_frame=obs_frame,
                contact_frame=contact_frame,
                language=language if language else None,
            )]

        ego4d_match = self._ego4d_contact_index.get(ep_stem)

        if self._ego4d_contact_index:
            if ego4d_match is None:
                logger.debug("No Ego4D contact match for %s, skipping", ep_stem)
                return []
            contact_frame = min(ego4d_match["contact_frame_local"], T - 1)
            # Use Ego4D narration as language if VITRA text is empty
            if not language and ego4d_match.get("narration"):
                language = ego4d_match["narration"]

        return [Interval(
            start=0,
            end=T - 1,
            obs_frame=obs_frame,
            contact_frame=contact_frame,
            language=language if language else None,
        )]

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """RGB frame from the source video; depth is always None.

        If the source video is not found, a placeholder image is returned.
        """
        video_name = episode_data.extra["video_name"]
        decode_frames = episode_data.extra["video_decode_frame"]
        subset = episode_data.extra["subset"]

        if frame_idx < 0 or frame_idx >= len(decode_frames):
            raise IndexError(
                f"frame_idx={frame_idx} out of range [0, {len(decode_frames)})"
            )

        video_frame_idx = int(decode_frames[frame_idx])

        video_path = self._resolve_video_path(video_name, subset)
        if video_path is not None:
            rgb = self._load_video_frame(video_path, video_frame_idx, episode_data)
        else:
            rgb = self._make_placeholder_frame(
                episode_data, frame_idx, video_frame_idx, video_name, subset
            )
        return rgb, None

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Active hand's ``tip_joint`` position in world frame (meters)."""
        hand_data = episode_data.extra["hand_data"]
        joints_w = np.asarray(hand_data["joints_worldspace"])  # (T, 21, 3)
        return joints_w[frame_idx, self.tip_joint, :].astype(np.float64)

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Active hand's wrist position in world frame (meters)."""
        hand_data = episode_data.extra["hand_data"]
        wrist = np.asarray(hand_data["transl_worldspace"])  # (T, 3)
        return wrist[frame_idx].astype(np.float64)

    def close_episode(self, episode_data: EpisodeData) -> None:
        """Close the cached PyAV container."""
        container = episode_data.extra.get("_av_container")
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        # Clear all pyav reader state
        for key in ("_av_container", "_av_container_path", "_av_stream",
                    "_av_fps", "_av_last_frame"):
            episode_data.extra.pop(key, None)

    # ------------------------------------------------------------------ #
    # Override: reproject moving-camera points to obs_frame
    # ------------------------------------------------------------------ #

    def reproject_camera_point_to_obs_frame(
        self,
        episode_data: EpisodeData,
        point_cam: np.ndarray,
        frame_idx: int,
        obs_frame_idx: int,
    ) -> np.ndarray:
        """Transform a point from frame_idx's camera frame to obs_frame's camera frame.

        Mainprocess back-projects each frame's mask with that frame's depth,
        so points start in that frame's own camera; this re-expresses them in
        the obs_frame camera using the per-frame SLAM extrinsics::

            P_world    = inv(T_world_to_cam_t) @ P_cam_t
            P_cam_obs  = T_world_to_cam_obs @ P_world
                       = T_world_to_cam_obs @ inv(T_world_to_cam_t) @ P_cam_t

        Frame indices are local episode frames (0..num_frames-1).
        """
        point_cam = np.asarray(point_cam, dtype=np.float64)
        if frame_idx == obs_frame_idx:
            return point_cam

        per_frame_ext = episode_data.extra.get("per_frame_extrinsics")
        if per_frame_ext is None:
            return point_cam

        n = len(per_frame_ext)
        local_t = min(max(int(frame_idx), 0), n - 1)
        local_obs = min(max(int(obs_frame_idx), 0), n - 1)

        if local_t == local_obs:
            return point_cam

        T_w2c_t = np.asarray(per_frame_ext[local_t], dtype=np.float64)
        T_w2c_obs = np.asarray(per_frame_ext[local_obs], dtype=np.float64)

        # P_cam_obs = T_w2c_obs @ inv(T_w2c_t) @ P_cam_t
        T_c_t_to_w = np.linalg.inv(T_w2c_t)
        T_combined = T_w2c_obs @ T_c_t_to_w  # (4, 4)

        p_hom = np.append(point_cam, 1.0)
        p_obs = (T_combined @ p_hom)[:3]
        return p_obs

    # ------------------------------------------------------------------ #
    # Override: compute_trajectory with per-frame extrinsics
    # ------------------------------------------------------------------ #

    def compute_trajectory(
        self, episode_data: EpisodeData, interval: Interval,
        cam_info: CameraInfo,
    ) -> List[TrajectoryPoint]:
        """Hand trajectory in the obs_frame camera frame (moving camera)."""
        # Use obs_frame extrinsic for all projections (consistent with obs image)
        obs_idx = interval.obs_frame if interval.obs_frame is not None else interval.start
        per_frame_ext = episode_data.extra["per_frame_extrinsics"]
        T_obs = per_frame_ext[obs_idx]  # (4, 4) world-to-camera at obs_frame

        pts: List[TrajectoryPoint] = []
        for idx in range(interval.start, interval.end + 1):
            tip_world = self.get_gripper_tip_3d(episode_data, idx)
            tip_cam = self.transform_point_to_camera(tip_world, T_obs)
            pixel = self.project_to_image(tip_cam, cam_info.intrinsics)
            pts.append(TrajectoryPoint(
                frame_idx=idx,
                position_3d=tip_cam.tolist(),
                position_pixel=list(pixel) if pixel else None,
            ))
        return pts

    def compute_contact_flange_point(
        self, episode_data: EpisodeData, contact_frame_idx: int,
        cam: CameraInfo,
    ) -> TrajectoryPoint:
        """Wrist at the contact frame, in the contact frame's own camera frame."""
        per_frame_ext = episode_data.extra["per_frame_extrinsics"]
        T_contact = per_frame_ext[contact_frame_idx]

        flange_world = self.get_flange_3d(episode_data, contact_frame_idx)
        flange_cam = self.transform_point_to_camera(flange_world, T_contact)
        flange_pixel = self.project_to_image(flange_cam, cam.intrinsics)
        return TrajectoryPoint(
            frame_idx=contact_frame_idx,
            position_3d=flange_cam.tolist(),
            position_pixel=list(flange_pixel) if flange_pixel else None,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_subset(episode_stem: str) -> str:
        """Determine source subset from episode filename prefix."""
        stem_lower = episode_stem.lower()
        if stem_lower.startswith("ego4d_"):
            # ego4d_cooking_and_cleaning and ego4d_other share the Ego4D videos
            return "ego4d"
        elif stem_lower.startswith("epic_"):
            return "epic"
        return "unknown"

    def _resolve_video_path(self, video_name: str, subset: str) -> Optional[Path]:
        """Find the source video; the first existing candidate wins.

        Directories tried: {video_root}/P{XX}/videos (EPIC only),
        {video_root}/{subset}_undistorted, {video_root}/{subset},
        {video_root}/{subset}/{video_name}, {video_root}.
        """
        extensions = [".mp4", ".MP4", ".webm", ".avi"]
        search_dirs = [
            self.video_root / f"{subset}_undistorted",  # undistorted preferred
            self.video_root / subset,
            self.video_root / subset / video_name,
            self.video_root,
        ]
        # EPIC-Kitchens real layout: {root}/P{XX}/videos/P{XX}_{YY}.MP4
        if subset == "epic":
            pid = video_name.split("_")[0]  # P01_01 -> P01
            search_dirs.insert(0, self.video_root / pid / "videos")

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            for ext in extensions:
                candidate = search_dir / f"{video_name}{ext}"
                if candidate.exists():
                    return candidate

        return None

    @staticmethod
    def _make_placeholder_frame(
        episode_data: EpisodeData, frame_idx: int,
        video_frame_idx: int, video_name: str, subset: str,
    ) -> np.ndarray:
        """480x640 dark image with episode metadata, used when the source video is missing."""
        import cv2
        H, W = 480, 640
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (30, 30, 50)  # dark blue-gray

        lines = [
            "VITRA placeholder (no video_root)",
            f"video: {video_name}",
            f"subset: {subset}",
            f"episode_frame: {frame_idx}/{episode_data.num_frames}",
            f"video_frame: {video_frame_idx}",
            f"lang: {episode_data.extra.get('language', '')[:50]}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img

    def _load_video_frame(
        self, video_path: Path, video_frame_idx: int, episode_data: EpisodeData,
    ) -> np.ndarray:
        """Load a single RGB frame from source video using PyAV.

        Seeks to the nearest keyframe before the target frame, then
        decodes forward until the target is reached. Caches the
        container and last-decoded position in episode_data.extra so
        sequential reads reuse the stream.
        """
        import av

        # load container (reuse across frames in same episode)
        container = episode_data.extra.get("_av_container")
        cached_path = episode_data.extra.get("_av_container_path")
        stream = episode_data.extra.get("_av_stream")
        fps = episode_data.extra.get("_av_fps")

        need_reopen = (
            container is None
            or cached_path != str(video_path)
        )
        if need_reopen:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 30.0
            episode_data.extra["_av_container"] = container
            episode_data.extra["_av_container_path"] = str(video_path)
            episode_data.extra["_av_stream"] = stream
            episode_data.extra["_av_fps"] = fps
            episode_data.extra["_av_last_frame"] = -1

        last_frame = episode_data.extra.get("_av_last_frame", -1)

        # For backward seeks OR initial read, do a seek
        if video_frame_idx <= last_frame or last_frame < 0:
            # Seek to nearest keyframe before target (in pts units)
            target_sec = video_frame_idx / fps
            seek_pts = int(target_sec / stream.time_base)
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
            episode_data.extra["_av_last_frame"] = -1

        # Decode forward until the first frame at or after the target; the
        # frame index is pts * fps rounded to the nearest integer.
        target_rgb = None
        best_frame = -1
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            cur_frame = int(float(frame.pts * stream.time_base) * fps + 0.5)
            episode_data.extra["_av_last_frame"] = cur_frame
            best_frame = cur_frame
            if cur_frame >= video_frame_idx:
                target_rgb = frame.to_ndarray(format="rgb24")
                break

        if target_rgb is None:
            raise IndexError(
                f"Could not decode frame {video_frame_idx} from {video_path} "
                f"(last decoded: {best_frame})"
            )

        # Handle unusual channel counts
        if target_rgb.ndim == 2:
            target_rgb = np.stack([target_rgb] * 3, axis=-1)
        elif target_rgb.shape[2] == 1:
            target_rgb = np.repeat(target_rgb, 3, axis=2)
        elif target_rgb.shape[2] == 4:
            target_rgb = target_rgb[:, :, :3]

        return target_rgb.astype(np.uint8)
