"""Loaders for the raw SceneFun3D layout: <root>/train_val/<visit>/<video>/..."""
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData


def _visit_json(visit_dir: Path, visit_id: str, name: str) -> list:
    with open(visit_dir / f"{visit_id}_{name}.json") as f:
        return json.load(f)[name]


def load_motions(visit_dir: Path, visit_id: str) -> list:
    return _visit_json(visit_dir, visit_id, "motions")


def load_annotations(visit_dir: Path, visit_id: str) -> list:
    return _visit_json(visit_dir, visit_id, "annotations")


def load_descriptions(visit_dir: Path, visit_id: str) -> list:
    """Entries with desc_id, annot_id (list) and description (task text)."""
    return _visit_json(visit_dir, visit_id, "descriptions")


def load_laser_scan(visit_dir: Path, visit_id: str) -> np.ndarray:
    """Faro laser scan -> (N, 3) float64 world points (annotations index into it)."""
    v = PlyData.read(str(visit_dir / f"{visit_id}_laser_scan.ply"))["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)


def load_frame_color(video_dir: Path, video_id: str, ts: str) -> np.ndarray:
    """hires_wide/<video>_<ts>.jpg -> HxWx3 uint8 BGR."""
    path = video_dir / "hires_wide" / f"{video_id}_{ts}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read frame: {path}")
    return img


def load_depth_raw(video_dir: Path, video_id: str, ts: str):
    """hires_depth/<video>_<ts>.png (uint16 mm, rendered from the laser scan), or None."""
    return cv2.imread(str(video_dir / "hires_depth" / f"{video_id}_{ts}.png"), cv2.IMREAD_UNCHANGED)


def load_frame_depth(video_dir: Path, video_id: str, ts: str) -> np.ndarray:
    """Depth as float32 meters."""
    depth = load_depth_raw(video_dir, video_id, ts)
    if depth is None:
        raise FileNotFoundError(f"could not read depth: {video_dir}/hires_depth/{video_id}_{ts}.png")
    return depth.astype(np.float32) / 1000.0


def load_intrinsics(video_dir: Path, video_id: str, ts: str) -> dict:
    """hires_wide_intrinsics/<video>_<ts>.pincam ("w h fx fy cx cy") -> dict."""
    with open(video_dir / "hires_wide_intrinsics" / f"{video_id}_{ts}.pincam") as f:
        w, h, fx, fy, cx, cy = f.read().split()
    return {"w": int(float(w)), "h": int(float(h)),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def load_poses(video_dir: Path) -> dict:
    """hires_poses.traj lines "ts rx ry rz tx ty tz" (axis-angle) -> {ts: (R, t)}.

    The direction (world->camera or camera->world) is resolved per video by alignment.
    """
    poses = {}
    with open(video_dir / "hires_poses.traj") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rvec = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            t = np.array([float(x) for x in parts[4:7]], dtype=np.float64)
            poses[parts[0]] = (cv2.Rodrigues(rvec)[0], t)
    return poses
