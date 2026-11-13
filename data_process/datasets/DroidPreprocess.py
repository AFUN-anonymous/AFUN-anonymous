"""
DROID preprocessing.

Raw data layout::

    {root}/<lab>/<success|failure>/<date>/<episode>/
        trajectory.h5            # robot_state/{cartesian_position, gripper_position}, camera_extrinsics
        metadata_<uuid>.json     # episode UUID (and camera serials)
        ext{1,2}_cam_serial      # serial numbers of the two external cameras
        recordings/SVO/<serial>.svo, recordings/MP4/<serial>.mp4

    {patch_root}/                # DROID annotation patch
        cam2base_extrinsics.json, cam2base_extrinsic_superset.json,
        cam2cam_extrinsics.json, episode_id_to_path.json, intrinsics.json,
        droid_language_annotations.json

Episodes are listed from episode_id_to_path.json; paths containing "failure"
are dropped. RGB-D comes from the left view of each external ZED camera (SVO,
needs the ZED SDK); RGB-only reads prefer the MP4. Intervals are the spans where
gripper_position > gripper_threshold. Extrinsics fall back per camera:
cam2base -> superset -> derived from the other camera via cam2cam -> trajectory.h5.
The tip is the flange plus gripper_offset_m along the flange z-axis.
"""

import os
import json
import logging
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

try:
    from datasets.utils.svo_reader import SVOReader
except ImportError:
    SVOReader = None  # ZED SDK unavailable; SVO reading will fail at runtime
from preprocess.base import RoboDatasetBase
from preprocess.dataclass import Interval, CameraInfo, EpisodeData


# ================================================================== #
# DROID-specific helpers
# ================================================================== #

def _6d_to_T(ext_6d) -> np.ndarray:
    """[x, y, z, rx, ry, rz] (Euler 'xyz') -> 4by4 transformation matrix."""
    pos = np.asarray(ext_6d[:3])
    rot = R.from_euler('xyz', ext_6d[3:6]).as_matrix()
    Transform = np.eye(4)
    Transform[:3, :3] = rot
    Transform[:3, 3] = pos
    return Transform


def _compute_gripper_tip(
    cartesian: np.ndarray, offset_m: float
) -> np.ndarray:
    """Compute gripper tip position from flange pose in base frame."""
    pos = cartesian[:3]
    rot = R.from_euler('xyz', cartesian[3:]).as_matrix()
    return pos + offset_m * rot[:, 2]


def _detect_gripper_closed_intervals(
    gripper_pos: np.ndarray, threshold: float, min_len: int
) -> List[Interval]:
    """Detect contiguous intervals where gripper_pos > threshold."""
    closed = gripper_pos > threshold
    intervals: List[Interval] = []
    start = None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start >= min_len:
                intervals.append(Interval(start=start, end=i - 1))
            start = None
    if start is not None and len(closed) - start >= min_len:
        intervals.append(Interval(start=start, end=len(closed) - 1))
    return intervals


# ================================================================== #
# DROID patch extrinsics loader
# ================================================================== #

class _PatchCameraLoader:
    """
    Loads DROID annotation-patch JSON files once and provides
    episode-level cam extrinsics / intrinsics lookup.

    Extrinsic fallback chain, per camera:
      1. cam2base_extrinsics.json
      2. cam2base_extrinsic_superset.json
      3. cam2cam_extrinsics.json  (derived from the other camera)
      4. Original h5 extrinsics
    """

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)

        print(f"[Droid PatchCameraLoader] Loading from {patch_dir} ...")
        self.cam2base = self._load("cam2base_extrinsics.json")
        self.superset = self._load("cam2base_extrinsic_superset.json")
        self.cam2cam = self._load("cam2cam_extrinsics.json")
        self.ep_id_map = self._load("episode_id_to_path.json")
        self.intrinsics = self._load("intrinsics.json")
        print("[Droid PatchCameraLoader] Ready.\n")

    def _load(self, name: str):
        with open(self.patch_dir / name) as f:
            return json.load(f)

    def get_episode_id(self, episode_path: str) -> str:
        """Return the episode UUID from the metadata_<uuid>.json file name."""
        for mf in Path(episode_path).glob("metadata_*.json"):
            return mf.stem.replace("metadata_", "", 1)
        raise FileNotFoundError(f"metadata file not found for path: {episode_path}")

    def get_intrinsics(
        self, episode_path: str, serial: str
    ) -> Optional[Dict[str, Any]]:
        """Return patch intrinsics {fx, fy, cx, cy, distCoeffs, resolution}, or None."""
        ep_id = self.get_episode_id(episode_path)
        ci = (self.intrinsics.get(ep_id) or {}).get(serial)
        if ci is None:
            return None
        cm = ci['cameraMatrix']  # [fx, cx, fy, cy]
        return {
            'fx': cm[0], 'fy': cm[2], 'cx': cm[1], 'cy': cm[3],
            'distCoeffs': ci['distCoeffs'],
            'resolution': (ci['width'], ci['height']),
        }

    def get_extrinsics(
        self,
        episode_path: str,
        ext1_serial: str,
        ext2_serial: str,
        ext1_h5: np.ndarray,
        ext2_h5: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Best-available T_base_to_cam for both cameras.

        Returns:
            (T_b2c_ext1, T_b2c_ext2, source_ext1, source_ext2)
        """
        ep_id = self.get_episode_id(episode_path)

        # Step 1-2: cam2base / superset per camera
        info: Dict[str, Dict[str, Any]] = {}
        for serial, label in [(ext1_serial, "ext1"), (ext2_serial, "ext2")]:
            T_c2b, src = None, None
            c2b = self.cam2base.get(ep_id, {})
            if serial in c2b:
                # Patch JSON stores T_cam_to_base; kept in that form here so
                # the cam2cam derivation step below can use it directly, and
                # inverted to T_base_to_cam at the end.
                T_c2b = _6d_to_T(c2b[serial])
                src = "cam2base"
            if src is None:
                ss = self.superset.get(ep_id, {})
                if serial in ss:
                    T_c2b = _6d_to_T(ss[serial])
                    src = "superset"
            info[label] = {'T_c2b': T_c2b, 'src': src}

        # Step 3: cam2cam derivation for the missing camera
        if ep_id in self.cam2cam:
            c2c = self.cam2cam[ep_id]
            T_e1r = np.array(c2c['left_cam']['pose'])
            T_e2r = np.array(c2c['right_cam']['pose'])
            if info['ext1']['src'] and not info['ext2']['src']:
                T_rel = np.linalg.inv(T_e1r) @ T_e2r # T_ext2_to_ext1 = T_relative_to_ext1 @ T_ext2_to_relative
                info['ext2']['T_c2b'] = info['ext1']['T_c2b'] @ T_rel # T_ext2_to_base = T_ext1_to_base @ T_ext2_to_ext1
                info['ext2']['src'] = "cam2cam_derived"
            elif info['ext2']['src'] and not info['ext1']['src']:
                T_rel = np.linalg.inv(T_e2r) @ T_e1r
                info['ext1']['T_c2b'] = info['ext2']['T_c2b'] @ T_rel
                info['ext1']['src'] = "cam2cam_derived"

        # Step 4: fallback to h5, then invert T_c2b -> T_b2c
        results = []
        for label, h5 in [("ext1", ext1_h5), ("ext2", ext2_h5)]:
            r = info[label]
            if r['src'] is None:
                # h5 6D is T_cam_to_base (cam pose in base); invert to get T_base_to_cam
                results.append((np.linalg.inv(_6d_to_T(h5)), "h5"))
            else:
                # patch T_c2b already computed; invert to get T_base_to_cam
                results.append((np.linalg.inv(r['T_c2b']), r['src']))

        print(f"  [PatchLoader] ext1: {results[0][1]}, ext2: {results[1][1]}")
        return results[0][0], results[1][0], results[0][1], results[1][1]


# ================================================================== #
# DroidPreprocess
# ================================================================== #

class DroidPreprocess(RoboDatasetBase):
    """DROID dataset preprocessor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        dataset = cfg.dataset
        self.patch_loader = _PatchCameraLoader(dataset.patch_root)
        self.gripper_threshold = dataset.gripper_threshold
        self.gripper_offset_m = dataset.gripper_offset_m
        self.min_interval_time = dataset.min_interval_time
        self._extract_depth = dataset.extract_depth

        # Language annotations  {uuid: {language_instruction1: ..., ...}}
        lang_path = Path(dataset.patch_root) / "droid_language_annotations.json"
        if lang_path.exists():
            with open(lang_path) as f:
                self._lang_ann = json.load(f)
            print(f"[Droid] Loaded {len(self._lang_ann)} language annotations.")
        else:
            self._lang_ann: dict = {}
            print(f"[Droid] Language annotations not found at {lang_path}.")

    # ------------------------------------------------------------------ #
    # Abstract implementations
    # ------------------------------------------------------------------ #

    def discover_episodes(self) -> List[str]:
        """Return every episode known to the patch loader, excluding any
        whose path contains 'failure'.
        """
        root = self.cfg.dataset.root
        all_episodes = sorted(
            os.path.join(root, p)
            for p in self.patch_loader.ep_id_map.values() if "failure" not in p
        )
        return all_episodes

    def get_episode_id(self, episode_path: str) -> str:
        """Return the episode UUID parsed from the metadata file."""
        uuid = self.patch_loader.get_episode_id(episode_path)
        if uuid:
            return uuid
        raise FileNotFoundError(f"Episode ID not found for path: {episode_path}. Check the metadata json file in the raw episode folder.")

    def load_episode(self, episode_path: str) -> EpisodeData:
        ep = Path(episode_path)
        h5_path = ep / "trajectory.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"trajectory.h5 not found in {episode_path}")

        # trajectory data
        with h5py.File(h5_path, 'r') as f:
            rs = f['observation']['robot_state']
            cart = rs['cartesian_position'][:]
            grip = rs['gripper_position'][:]

        # camera serials
        def _serial(name: str) -> Optional[str]:
            p = ep / name
            return p.read_text().strip() if p.exists() else None

        s1, s2 = _serial("ext1_cam_serial"), _serial("ext2_cam_serial")
        if not s1 or not s2:
            for mf in ep.glob("metadata_*.json"):
                m = json.loads(mf.read_text())
                s1 = s1 or m.get("ext1_cam_serial")
                s2 = s2 or m.get("ext2_cam_serial")
                break
        if not s1 or not s2:
            raise ValueError(
                f"Camera serials not found: ext1={s1}, ext2={s2}"
            )

        # h5 extrinsics (frame 0, used as fallback)
        with h5py.File(h5_path, 'r') as f:
            ce = f['observation']['camera_extrinsics']
            h5_ext1 = ce[f"{s1}_left"][0]
            h5_ext2 = ce[f"{s2}_left"][0]

        # patched extrinsics
        T_ext1, T_ext2, src1, src2 = self.patch_loader.get_extrinsics(
            episode_path, s1, s2, h5_ext1, h5_ext2
        )

        # camera infos (open SVO readers for video, load patch intrinsics)
        svo_dir = ep / "recordings" / "SVO"
        mp4_dir = ep / "recordings" / "MP4"
        cams: List[CameraInfo] = []
        svo_readers: Dict[str, SVOReader] = {}
        mp4_paths: Dict[str, str] = {}

        for serial, T_b2c, src in [(s1, T_ext1, src1), (s2, T_ext2, src2)]:
            if self._extract_depth:
                # Need SVO for depth
                if SVOReader is None:
                    raise ImportError(
                        "extract_depth=True requires the ZED SDK (pyzed) "
                        "which is not available"
                    )
                svo_path = svo_dir / f"{serial}.svo"
                if not svo_path.exists():
                    raise FileNotFoundError(f"SVO not found: {svo_path}")
                reader = SVOReader(str(svo_path), depth_mode="NEURAL")
                svo_readers[serial] = reader
            else:
                # RGB-only: prefer MP4 (no ZED SDK needed), fall back to SVO
                mp4_path = mp4_dir / f"{serial}.mp4"
                if mp4_path.exists():
                    mp4_paths[serial] = str(mp4_path)
                else:
                    svo_path = svo_dir / f"{serial}.svo"
                    if not svo_path.exists():
                        raise FileNotFoundError(
                            f"Neither MP4 nor SVO found for {serial} in {ep}"
                        )
                    if SVOReader is None:
                        raise ImportError(
                            f"MP4 not found for {serial} and ZED SDK (pyzed) "
                            f"is unavailable for SVO fallback in {ep}"
                        )
                    reader = SVOReader(str(svo_path), depth_mode="NONE")
                    svo_readers[serial] = reader

            intr = self.patch_loader.get_intrinsics(episode_path, serial)
            if not intr or intr.get('fx', 0) == 0 or intr.get('fy', 0) == 0:
                raise ValueError(
                    f"Patch intrinsics not found or invalid for serial {serial} "
                    f"in {episode_path}"
                )
            cams.append(CameraInfo(
                cam_id=serial,
                intrinsics=intr,
                T_base_to_cam=T_b2c,
                extrinsic_source=src,
                intrinsic_source="patch",
                distortion=None,
            ))

        if svo_readers:
            fps = next(iter(svo_readers.values())).get_fps()
        elif mp4_paths:
            import cv2
            cap = cv2.VideoCapture(next(iter(mp4_paths.values())))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            fps = None

        if not fps:
            logger.warning(
                "Could not determine FPS for episode %s; skipping",
                ep,
            )
            raise ValueError(f"FPS unavailable for episode {episode_path}")

        uuid = self.patch_loader.get_episode_id(episode_path)

        return EpisodeData(
            num_frames=len(cart),
            fps=fps,
            camera_infos=cams,
            extra={
                'uuid': uuid,
                'cartesian': cart,
                'gripper': grip,
                'svo_readers': svo_readers,
                'mp4_paths': mp4_paths,
            },
        )

    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        intervals = _detect_gripper_closed_intervals(
            episode_data.extra['gripper'],
            self.gripper_threshold,
            int(self.min_interval_time * episode_data.fps),
        )

        # Attach language + obs_frame to each interval
        uuid = episode_data.extra.get('uuid')
        lang = self._get_longest_language(uuid) if uuid else None

        for i, iv in enumerate(intervals):
            iv.language = lang
            if i == 0:
                iv.obs_frame = 0  # first interval: episode frame 0
            # later intervals: obs_frame stays None, i.e. iv.start

        return intervals

    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        reader = episode_data.extra['svo_readers'][cam_id]
        return reader.read_frame(frame_idx, get_depth=self._extract_depth)

    def extract_frame(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> np.ndarray:
        """RGB-only extraction -- uses MP4 via OpenCV when available."""
        mp4_path = episode_data.extra.get('mp4_paths', {}).get(cam_id)
        if mp4_path is not None:
            import cv2
            # Lazy-open a VideoCapture and cache it
            caps = episode_data.extra.setdefault('_mp4_caps', {})
            cap = caps.get(cam_id)
            if cap is None:
                cap = cv2.VideoCapture(mp4_path)
                caps[cam_id] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {mp4_path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Fall back to SVO reader
        rgb, _ = self.extract_frame_and_depth(episode_data, cam_id, frame_idx)
        return rgb

    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        return _compute_gripper_tip(
            episode_data.extra['cartesian'][frame_idx], self.gripper_offset_m
        )

    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        return episode_data.extra['cartesian'][frame_idx][:3].copy()

    def close_episode(self, episode_data: EpisodeData) -> None:
        for r in episode_data.extra.get('svo_readers', {}).values():
            r.close()
        for cap in episode_data.extra.get('_mp4_caps', {}).values():
            cap.release()

    # ---- language helper ------------------------------------------------ #

    def _get_longest_language(self, uuid: str) -> Optional[str]:
        """Return the longest language instruction for *uuid*, or None."""
        ann = self._lang_ann.get(uuid)
        if not ann:
            return None
        texts = [v for v in ann.values() if isinstance(v, str) and v.strip()]
        return max(texts, key=len) if texts else None
