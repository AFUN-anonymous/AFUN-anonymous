"""Abstract base class for dataset preprocessing adapters."""

from abc import ABC, abstractmethod
import json
import shutil
import sys
import contextlib
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import imageio
import numpy as np
import ray
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from preprocess.dataclass import CameraInfo, EpisodeData, Interval, Motion, TrajectoryPoint


class RoboDatasetBase(ABC):
    """Shared preprocess pipeline: load episode, detect intervals, export each camera view.

    Subclasses implement dataset-specific loading and interval detection.

    Output layout::

        {output_root}/{dataset_name}/{episode_id}/interval_{NNN}/cam_{N}/

    Units: depth in millimeters, positions in meters, extrinsics as T_base_to_cam.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.output_root = Path(cfg.output_root) / cfg.dataset.name
        if cfg.get('multi_processing'):
            import torch
            n_gpus = torch.cuda.device_count()
            print(f"[MP - preprocess] Using {n_gpus} GPUs")
            ray.init(
                ignore_reinit_error=True,
                num_cpus=cfg.num_processes,
                num_gpus=n_gpus,
            )
            print(f"[MP - preprocess] Using {cfg.num_processes} workers")

    # ------------------------------------------------------------------ #
    # Abstract (dataset-specific)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def discover_episodes(self) -> List[str]:
        """Return all episode paths to process.

        Used when ``cfg.dataset.episode_txt_list`` is null. Each returned string
        is passed to ``get_episode_id`` and ``load_episode``.
        """
        ...

    @abstractmethod
    def get_episode_id(self, episode_path: str) -> str:
        """Return a filesystem-safe episode ID, unique across all datasets.

        It names the output folder and is written to metadata.json and
        episode_summary.json.
        """
        ...

    @abstractmethod
    def load_episode(self, episode_path: str) -> EpisodeData:
        """Load one episode into an EpisodeData."""
        ...

    @abstractmethod
    def detect_intervals(self, episode_data: EpisodeData) -> List[Interval]:
        """Return the manipulation intervals of the episode."""
        ...

    @abstractmethod
    def extract_frame_and_depth(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (RGB uint8 HxWx3, depth or None).

        Depth must be float32 in millimeters.
        """
        ...

    def extract_frame(
        self, episode_data: EpisodeData, cam_id: str, frame_idx: int
    ) -> np.ndarray:
        """Return RGB only. Override when RGB can be read without depth."""
        rgb, _ = self.extract_frame_and_depth(episode_data, cam_id, frame_idx)
        return rgb

    @abstractmethod
    def get_gripper_tip_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Gripper tip position (3,) in the robot base (world) frame, meters."""
        ...

    @abstractmethod
    def get_flange_3d(
        self, episode_data: EpisodeData, frame_idx: int
    ) -> np.ndarray:
        """Flange position (3,) in the robot base (world) frame, meters (no tip offset)."""
        ...

    @abstractmethod
    def close_episode(self, episode_data: EpisodeData) -> None:
        """Release any resources held for this episode."""
        ...

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def transform_point_to_camera(
        point_base: np.ndarray, T_base_to_cam: np.ndarray
    ) -> np.ndarray:
        """Transform a 3D point from robot base frame to camera frame."""
        return (T_base_to_cam @ np.append(point_base, 1.0))[:3]

    @staticmethod
    def project_to_image(
        point_3d: np.ndarray, intrinsics: Dict[str, Any]
    ) -> Optional[Tuple[int, int]]:
        """Project a 3D point in camera frame to image pixel coordinates."""
        x, y, z = point_3d
        if z <= 0:
            return None
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        return (int(round(fx * x / z + cx)), int(round(fy * y / z + cy)))

    def reproject_camera_point_to_obs_frame(
        self,
        episode_data: EpisodeData,
        point_cam: np.ndarray,
        frame_idx: int,
        obs_frame_idx: int,
    ) -> np.ndarray:
        """Move a point from frame_idx's camera frame into obs_frame's camera frame.

        Identity for static cameras. Moving-camera datasets must override this.
        Used by mainprocess (sam3_mask_traj, sam3_mask_traj_depth).
        """
        return np.asarray(point_cam, dtype=np.float64)

    @staticmethod
    def draw_trajectory_on_image(
        image: np.ndarray,
        trajectory: List[TrajectoryPoint],
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 3,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw trajectory points and connecting lines on an image."""
        img = image.copy()
        pixels = [tuple(p.position_pixel) for p in trajectory if p.position_pixel]
        for i in range(len(pixels) - 1):
            cv2.line(img, pixels[i], pixels[i + 1], color, thickness)
        for px in pixels:
            cv2.circle(img, px, radius, color, -1)
        if pixels:
            cv2.circle(img, pixels[0], radius + 2, (255, 0, 0), 2)   # start
            cv2.circle(img, pixels[-1], radius + 2, (0, 0, 255), 2)  # end
        return img

    # ------------------------------------------------------------------ #
    # Trajectory / motion
    # ------------------------------------------------------------------ #

    def compute_trajectory(
        self, episode_data: EpisodeData, interval: Interval,
        cam_info: CameraInfo,
    ) -> List[TrajectoryPoint]:
        """Per-frame gripper tip trajectory in the camera frame (meters)."""
        pts: List[TrajectoryPoint] = []
        for idx in range(interval.start, interval.end + 1):
            tip_base = self.get_gripper_tip_3d(episode_data, idx)
            tip_cam = self.transform_point_to_camera(
                tip_base, cam_info.T_base_to_cam
            )
            pixel = self.project_to_image(tip_cam, cam_info.intrinsics)
            pts.append(TrajectoryPoint(
                frame_idx=idx,
                position_3d=tip_cam.tolist(),
                position_pixel=list(pixel) if pixel else None,
            ))
        return pts

    def compute_contact_flange_point(
        self, episode_data: EpisodeData, contact_frame_idx: int, cam: CameraInfo,
    ) -> TrajectoryPoint:
        """Flange position at the contact frame, in the camera frame (meters)."""
        flange_base = self.get_flange_3d(episode_data, contact_frame_idx)
        flange_cam = self.transform_point_to_camera(flange_base, cam.T_base_to_cam)
        flange_pixel = self.project_to_image(flange_cam, cam.intrinsics)
        return TrajectoryPoint(
            frame_idx=contact_frame_idx,
            position_3d=flange_cam.tolist(),
            position_pixel=list(flange_pixel) if flange_pixel else None,
        )

    def compute_motion(
        self, trajectory: List[TrajectoryPoint], fps: float,
    ) -> Motion:
        """Normalized 3D motion and 2D motion endpoints over the first ~1 s."""
        if len(trajectory) < int(fps) * 0.5:  # need at least 0.5 s of trajectory
            raise ValueError(
                f"Trajectory too short ({len(trajectory)} frames) for motion to be meaningful. "
                f"min_interval_time should be at least 0.5 s (need {int(fps * 0.5)} frames at {fps} fps)."
            )
        end_idx = min(max(1, int(fps)), len(trajectory) - 1)
        disp = (
            np.array(trajectory[end_idx].position_3d)
            - np.array(trajectory[0].position_3d)
        )
        norm = np.linalg.norm(disp)
        return Motion(
            motion_3d=(disp / norm).tolist() if norm > 1e-10 else [0.0, 0.0, 0.0],
            motion_2d_start=trajectory[0].position_pixel,
            motion_2d_end=trajectory[end_idx].position_pixel,
        )

    # ------------------------------------------------------------------ #
    # Writers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_images(cam_dir: Path, obs_frame, contact_frame, annotated) -> None:
        """Save obs_frame.png, contact_frame.png and obs_frame_traj.png."""
        imageio.imwrite(str(cam_dir / "obs_frame.png"), obs_frame)
        imageio.imwrite(str(cam_dir / "contact_frame.png"), contact_frame)
        imageio.imwrite(str(cam_dir / "obs_frame_traj.png"), annotated)

    @staticmethod
    def _save_depth_npy(cam_dir: Path, depth) -> str:
        """Save obs_frame_depth.npy (float32, mm). Returns the depth source tag."""
        if depth is not None:
            np.save(str(cam_dir / "obs_frame_depth.npy"), depth)
            return "dataset"
        return "none"

    @staticmethod
    def _save_traj_and_cam(cam_dir: Path, cam: CameraInfo, traj, motion,
                           contact_flange_point: Optional[TrajectoryPoint] = None) -> None:
        """Save trajectory.json: camera info, 3D/2D trajectory and motion."""
        data = {
            'camera_info': {
                'cam_id': cam.cam_id,
                'intrinsics': cam.intrinsics,
                'T_base_to_cam': cam.T_base_to_cam.tolist(),
                'extrinsic_source': cam.extrinsic_source,
                'intrinsic_source': cam.intrinsic_source,
                'distortion': cam.distortion,
            },
            'trajectory_3d': [
                {'frame_idx': p.frame_idx, 'position_3d': p.position_3d}
                for p in traj
            ],
            'trajectory_2d': [
                {'frame_idx': p.frame_idx,
                    'position_pixel': p.position_pixel}
                for p in traj
            ],
            'motion_3d': motion.motion_3d,
            'motion_2d': {
                'start_pixel': motion.motion_2d_start,
                'end_pixel': motion.motion_2d_end,
            },
        }
        if contact_flange_point is not None:
            data['contact_flange_point'] = asdict(contact_flange_point)
        with open(cam_dir / "trajectory.json", 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _collect_cam_info_for_interval_metadata(cam_meta: Dict[str, Any], ci: int, cam: CameraInfo, depth_src: str) -> None:
        cam_meta[f"cam_{ci}"] = {
            'cam_id': cam.cam_id,
            'extrinsic_source': cam.extrinsic_source,
            'intrinsic_source': cam.intrinsic_source,
            'depth_source': depth_src,
        }

    @staticmethod
    def _save_interval_metadata(interval_dir: Path, episode_id: str, i: int,
                                interval: Interval, cams: List[CameraInfo], cam_meta: dict) -> None:
        """Save metadata.json in interval_{NNN}/."""
        with open(interval_dir / "metadata.json", 'w') as f:
            json.dump({
                'episode_id': episode_id,
                'interval_idx': i,
                'start_frame': interval.start,
                'end_frame': interval.end,
                'obs_frame': interval.obs_frame,
                'contact_frame': interval.contact_frame,
                'interval_language': interval.language,
                'num_cams': len(cams),
                **cam_meta,
            }, f, indent=2)

        print(f"    interval_{i:03d}  [{interval.start} - {interval.end}]")

    @staticmethod
    def _save_episode_summary(episode_dir: Path, episode_id: str, episode_path: str,
                              episode_data: EpisodeData, intervals: List[Interval], cams: List[CameraInfo], fps: float) -> None:
        """Save episode_summary.json in the episode folder."""
        with open(episode_dir / "episode_summary.json", 'w') as f:
            json.dump({
                'episode_id': episode_id,
                'episode_path': str(episode_path),
                'num_frames': episode_data.num_frames,
                'num_intervals': len(intervals),
                'num_cams': len(cams),
                'fps': fps,
                'intervals': [
                    {'idx': j, 'start': iv.start, 'end': iv.end,
                     'length': iv.length}
                    for j, iv in enumerate(intervals)
                ],
            }, f, indent=2)

    # ------------------------------------------------------------------ #
    # Episode list
    # ------------------------------------------------------------------ #

    def _read_episode_list(self) -> Optional[List[str]]:
        """Read ``cfg.dataset.episode_txt_list`` (one episode path per line), or None if unset."""
        txt_path = self.cfg.dataset.get("episode_txt_list", None)
        if txt_path is None:
            return None

        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(
                f"episode_txt_list is set but file not found: {txt_path}"
            )

        with open(txt_path, "r") as f:
            episodes = [line.strip() for line in f if line.strip()]

        if not episodes:
            raise ValueError(f"episode_txt_list file is empty: {txt_path}. Please set config.dataset.episode_txt_list to null or provide a non-empty episode list file.")

        print(f"[list_episodes] Loaded {len(episodes)} episodes from {txt_path}")
        return episodes

    def list_episodes(self) -> List[str]:
        """Episodes from ``episode_txt_list`` if set, else ``discover_episodes()``."""
        eps = self._read_episode_list()
        if eps is not None:
            return eps
        return self.discover_episodes()

    # ------------------------------------------------------------------ #
    # Per-episode pipeline
    # ------------------------------------------------------------------ #

    def process_episode(self, episode_path: str, timestamp: str) -> Optional[Dict]:
        """Process one episode; on error, log it to log/{timestamp}_error.log and remove partial output."""
        try:
            return self._process_episode(episode_path)
        except Exception as e:
            print(f"Error processing episode {episode_path}: {e}, Skipping this episode.")

            traceback.print_exc()
            log_dir = Path("log")
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(log_dir / f"{timestamp}_error.log", "a") as log_file:
                log_file.write(f"Error processing episode {episode_path}: {e}\n")
                traceback.print_exc(file=log_file)

            try:
                episode_id = self.get_episode_id(episode_path)
                episode_dir = self.output_root / episode_id
                if episode_dir.exists():
                    shutil.rmtree(episode_dir)
                    print(f"  Cleaned up partial output: {episode_dir}")
            except Exception:
                pass  # get_episode_id itself may have failed; nothing to clean

            return None

    def _episode_outputs_complete(self, episode_dir: Path) -> bool:
        # episode_summary.json is the last file written on a successful run.
        if not (episode_dir / "episode_summary.json").exists():
            return False
        # With write_query_aux_frames, the aux frames must be present too.
        if bool(self.cfg.dataset.get("write_query_aux_frames", False)):
            canary_dir = episode_dir / "interval_000" / "cam_0"
            if not (canary_dir / "query_aux_frame_mid.png").exists():
                return False
            if not (canary_dir / "query_aux_frame_end.png").exists():
                return False
        return True

    def _process_episode(self, episode_path: str) -> Optional[Dict]:
        """Load the episode, detect intervals and write every interval/camera view."""
        # Check skip_existing before load_episode, which can be expensive.
        episode_id = self.get_episode_id(episode_path)
        episode_dir = self.output_root / episode_id
        if bool(self.cfg.get("skip_existing", True)) and \
                self._episode_outputs_complete(episode_dir):
            print(f"[skip_existing] {episode_id}")
            return {"episode_id": episode_id, "skipped": True}

        print(f"\n{'=' * 60}\n  {episode_path}\n{'=' * 60}")

        episode_data = self.load_episode(episode_path)
        intervals = self.detect_intervals(episode_data)

        if not intervals:
            print("  No intervals found, skipping.")
            self.close_episode(episode_data)
            return None

        fps = episode_data.fps
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

                traj = self.compute_trajectory(episode_data, interval, cam)
                motion = self.compute_motion(traj, fps)
                annotated = self.draw_trajectory_on_image(obs_frame, traj)

                contact_flange_pt = self.compute_contact_flange_point(
                    episode_data, contact_frame_idx, cam
                )

                self._save_images(cam_dir, obs_frame, contact_frame, annotated)

                # Middle and end frames of the interval, for the 3-frame query
                # generation prompt of datasets without contact annotations.
                if bool(self.cfg.dataset.get("write_query_aux_frames", False)):
                    T_iv = interval.end - interval.start + 1
                    mid_frame_idx = interval.start + T_iv // 2
                    end_frame_idx = interval.end
                    try:
                        mid_frame, _ = self.extract_frame_and_depth(
                            episode_data, cam.cam_id, mid_frame_idx
                        )
                        end_frame, _ = self.extract_frame_and_depth(
                            episode_data, cam.cam_id, end_frame_idx
                        )
                        imageio.imwrite(str(cam_dir / "query_aux_frame_mid.png"), mid_frame)
                        imageio.imwrite(str(cam_dir / "query_aux_frame_end.png"), end_frame)
                    except Exception as _aux_err:
                        print(f"[preprocess] query_aux_frame extraction failed "
                              f"for {cam_dir}: {_aux_err}")
                depth_src = self._save_depth_npy(cam_dir, depth)

                self._save_traj_and_cam(cam_dir, cam, traj, motion, contact_flange_pt)

                self._collect_cam_info_for_interval_metadata(cam_meta, ci, cam, depth_src)

            self._save_interval_metadata(interval_dir, episode_id, i, interval, cams, cam_meta)

        self._save_episode_summary(episode_dir, episode_id, episode_path, episode_data, intervals, cams, fps)

        self.close_episode(episode_data)
        print(f"Done: {episode_id}  ({len(intervals)} intervals)")
        return {'episode_id': episode_id, 'num_intervals': len(intervals)}

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(self):
        """Process all episodes, sequentially or with Ray workers."""
        episodes = self.list_episodes()
        episodes = episodes[:int(self.cfg['limit'])] if self.cfg.get('limit') is not None else episodes

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f"[run data preprocess] {len(episodes)} episodes -> {self.output_root}")
        results = []
        with contextlib.redirect_stdout(DummyTqdmFile(sys.stdout)):
            if self.cfg.get('multi_processing'):
                print(f"[run data preprocess] Processing {len(episodes)} episodes in parallel with {self.cfg.num_processes} workers.")

                num_workers = int(ray.available_resources().get("CPU", 1))
                num_workers = max(1, min(num_workers, len(episodes)))

                # Request a GPU share per worker only when the cluster has GPUs;
                # on CPU-only nodes a GPU request would never be scheduled.
                _cluster_gpus = int(ray.cluster_resources().get("GPU", 0) or 0)
                if _cluster_gpus > 0:
                    _worker_num_gpus = _cluster_gpus / self.cfg.num_processes
                else:
                    _worker_num_gpus = 0
                RemotePreproc = ray.remote(self.__class__).options(
                    num_cpus=1, num_gpus=_worker_num_gpus,
                )
                workers = [RemotePreproc.remote(self.cfg) for _ in range(num_workers)]

                # Bound the number of in-flight tasks so very large episode
                # lists do not exhaust the raylet's memory.
                max_in_flight = max(64, num_workers * 4)
                pending = {}  # ObjectRef -> ep
                pbar = tqdm(total=len(episodes), desc="Processing (ray)")

                def _drain_one():
                    done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
                    done_ref = done_refs[0]
                    ep_done = pending.pop(done_ref)
                    try:
                        r = ray.get(done_ref)
                        if r:
                            results.append(r)
                    except Exception as e:
                        tqdm.write(f"  ERROR: {ep_done}\n    {e}")
                    pbar.update(1)

                for i, ep in enumerate(episodes):
                    while len(pending) >= max_in_flight:
                        _drain_one()
                    fut = workers[i % num_workers].process_episode.remote(ep, timestamp)
                    pending[fut] = ep

                while pending:
                    _drain_one()
                pbar.close()

            else:
                print(f"[run data preprocess] Processing {len(episodes)} episodes sequentially.")
                for ep in tqdm(episodes, desc="Processing"):
                    try:
                        r = self.process_episode(ep, timestamp)
                        if r:
                            results.append(r)
                    except Exception as e:
                        tqdm.write(f"  ERROR: {ep}\n    {e}")
                        traceback.print_exc()
        print(f"\n[run data preprocess] Done. {len(results)}/{len(episodes)} processed.")
        return results
