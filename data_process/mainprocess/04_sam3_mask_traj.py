#!/usr/bin/env python3
"""
04_sam3_mask_traj.py - Object trajectory from SAM3 tracking masks + sensor depth.

For each tracked mask frame in [obs_frame, end_frame] (from stage 02.5):
  1. Read the depth map with the dataset preprocessor's extract_frame_and_depth.
  2. Back-project every mask pixel with valid depth to 3D (camera frame) and
     take the component-wise median as the object position.
  3. Express it in the obs_frame camera (identity for static cameras) and
     project it to obs_frame pixels for trajectory_2d.
Tracks whose start-to-end displacement is below min_start_end_dist_3d_m or
min_start_end_dist_2d_px are filtered out.

Inputs (per cam directory):
  sam3_text_prompt_tracking_masks/   binary mask PNGs (frame_NNNNN.png)
  sam3_mask_meta.json                {"stride"}
  trajectory.json                    camera_info (intrinsics, cam_id)

Outputs:
  sam3_mask_traj.json       {camera_info, trajectory_3d, trajectory_2d}
  obs_frame_sam3_traj.png   trajectory drawn on obs_frame
  {output_root}/sam3_mask_traj_manifest.json

Run:
  python mainprocess.py stage=sam3_mask_traj dataset_name=droid skip_existing=true
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .utils import (
    STAGE_SAM3_MASK_TRAJ,
    collect_episode_dirs,
    done_keys,
    get_dataset_preprocessor,
    get_dataset_roots,
    load_manifest,
    save_manifest,
)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _load_mask(path: Path) -> np.ndarray:
    """Load a binary mask (uint8 0/255) and return bool array."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 0


def _mask_to_3d(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: Dict[str, Any],
) -> Optional[Tuple[List[float], float, float]]:
    """Back-project all valid mask pixels to 3D, return component-wise median.

    Returns (pos_3d, cx_px, cy_px):
      - pos_3d: [x, y, z] in meters (camera frame), component-wise median
      - cx_px, cy_px: 2D pixel position (re-projection of pos_3d)

    Robust to depth outliers and non-convex masks.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    depths = depth_mm[ys, xs].astype(np.float64)
    valid = depths > 0
    if not np.any(valid):
        return None

    xv, yv, dv = xs[valid].astype(np.float64), ys[valid].astype(np.float64), depths[valid]

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    z = dv / 1000.0  # mm -> m
    x3d = (xv - cx) * z / fx
    y3d = (yv - cy) * z / fy

    pos_3d = [float(np.median(x3d)), float(np.median(y3d)), float(np.median(z))]

    # Re-project to 2D for trajectory_2d
    if pos_3d[2] > 0:
        cx_px = fx * pos_3d[0] / pos_3d[2] + cx
        cy_px = fy * pos_3d[1] / pos_3d[2] + cy
    else:
        cx_px, cy_px = float(np.median(xv)), float(np.median(yv))

    return pos_3d, float(cx_px), float(cy_px)


# ---------------------------------------------------------------------------
# Main processing logic
# ---------------------------------------------------------------------------

def _start_end_distance(points: List[Dict[str, Any]], key: str) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(
        np.asarray(points[-1][key], dtype=np.float32)
        - np.asarray(points[0][key], dtype=np.float32)
    ))


def compute_mask_trajectory(
    cam_dir: str,
    min_start_end_dist_3d_m: float,
    min_start_end_dist_2d_px: float,
    output_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute the object trajectory from SAM3 tracking masks + depth.

    Depth is read via the dataset preprocessor's extract_frame_and_depth,
    which handles each dataset's native depth format (SVO, PNG, etc.).

    Returns the dict that will be serialised to sam3_mask_traj.json.
    """
    cam_dir = Path(cam_dir)

    mask_meta = json.loads((cam_dir / "sam3_mask_meta.json").read_text())
    stride: int = mask_meta["stride"]
    interval_meta = json.loads((cam_dir.parent / "metadata.json").read_text())
    interval_start: int = interval_meta.get("obs_frame", interval_meta["start_frame"]) or 0
    interval_end: int = interval_meta["end_frame"]
    # Masks are indexed from obs_frame: frame_00000 = obs_frame,
    # frame_00001 = obs_frame + stride, ... The trajectory covers [obs_frame, end_frame].
    mask_start_frame: int = interval_meta.get("obs_frame", 0) or 0

    traj_data = json.loads((cam_dir / "trajectory.json").read_text())
    camera_info = traj_data["camera_info"]
    intrinsics = camera_info["intrinsics"]
    cam_serial = camera_info["cam_id"]

    episode_summary = json.loads(
        (cam_dir.parent.parent / "episode_summary.json").read_text()
    )
    episode_path = episode_summary["episode_path"]

    mask_files = sorted((cam_dir / "sam3_text_prompt_tracking_masks").glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {cam_dir}/sam3_text_prompt_tracking_masks")
    print(f"Found {len(mask_files)} mask frames (mask_start={mask_start_frame}, "
          f"interval=[{interval_start}, {interval_end}])")

    dataset_name = cam_dir.parent.parent.parent.name  # {output_root}/{dataset}/{ep}/interval/cam
    if output_root is None:
        output_root = cam_dir.parent.parent.parent.parent
    preprocessor = get_dataset_preprocessor(dataset_name, output_root)
    episode_data = preprocessor.load_episode(episode_path)
    print(f"Depth source: {dataset_name} preprocessor, cam '{cam_serial}'")

    trajectory_3d: List[Dict] = []
    trajectory_2d: List[Dict] = []
    skipped = 0

    try:
        for mask_path in mask_files:
            # Parse frame index from filename (frame_NNNNN.png), not enumerate position
            local_idx = int(mask_path.stem.split("_")[-1])
            video_frame_idx = mask_start_frame + local_idx * stride

            # Clip to [obs_frame, end_frame]
            if video_frame_idx < interval_start or video_frame_idx > interval_end:
                skipped += 1
                continue

            mask = _load_mask(mask_path)
            if not np.any(mask):
                skipped += 1
                continue

            try:
                _, depth_mm = preprocessor.extract_frame_and_depth(
                    episode_data, cam_serial, video_frame_idx
                )
            except Exception as e:
                print(f"  WARNING: depth read failed at frame {video_frame_idx}: {e}")
                skipped += 1
                continue

            if depth_mm is None:
                skipped += 1
                continue

            result = _mask_to_3d(mask, depth_mm.astype(np.float32), intrinsics)
            if result is None:
                skipped += 1
                continue

            pos_3d, cx_px, cy_px = result

            # Re-project to obs_frame's camera coordinate system. For static
            # cameras this is a no-op (base class default). For moving-camera
            # datasets (e.g. VITRA egocentric) the preprocessor overrides
            # reproject_camera_point_to_obs_frame() to apply the per-frame
            # extrinsic transformation. After transforming 3D, we re-project
            # to 2D using the obs_frame intrinsics so that trajectory_2d is
            # consistent with the obs_frame image.
            pos_3d_np = np.asarray(pos_3d, dtype=np.float64)
            pos_3d_obs = preprocessor.reproject_camera_point_to_obs_frame(
                episode_data, pos_3d_np, video_frame_idx, interval_start
            )
            if not np.allclose(pos_3d_obs, pos_3d_np, atol=1e-8):
                # Moving camera — re-project the transformed 3D point to 2D
                # in obs_frame's image
                z_obs = float(pos_3d_obs[2])
                if z_obs <= 0:
                    skipped += 1
                    continue
                fx, fy = intrinsics["fx"], intrinsics["fy"]
                cx_cam, cy_cam = intrinsics["cx"], intrinsics["cy"]
                cx_px = fx * float(pos_3d_obs[0]) / z_obs + cx_cam
                cy_px = fy * float(pos_3d_obs[1]) / z_obs + cy_cam

            trajectory_3d.append({"frame_idx": video_frame_idx,
                                   "position_3d": [float(x) for x in pos_3d_obs]})
            trajectory_2d.append({"frame_idx": video_frame_idx,
                                   "position_pixel": [round(cx_px, 2), round(cy_px, 2)]})

            n_processed = len(trajectory_3d) + skipped
            if n_processed % 50 == 0 or mask_path == mask_files[-1]:
                print(f"  processed {n_processed}/{len(mask_files)} masks (skipped {skipped})")
    finally:
        preprocessor.close_episode(episode_data)

    print(f"Done. {len(trajectory_3d)} valid trajectory points, {skipped} skipped.")

    dist_3d = _start_end_distance(trajectory_3d, "position_3d")
    dist_2d = _start_end_distance(trajectory_2d, "position_pixel")
    if dist_3d < min_start_end_dist_3d_m or dist_2d < min_start_end_dist_2d_px:
        print(
            f"Filtering near-static trajectory: "
            f"3D {dist_3d:.4f} m < {min_start_end_dist_3d_m:.4f} m or "
            f"2D {dist_2d:.2f} px < {min_start_end_dist_2d_px:.2f} px"
        )
        return None

    return {"camera_info": camera_info, "trajectory_3d": trajectory_3d, "trajectory_2d": trajectory_2d}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _project_to_image(point_3d: List[float], intrinsics: dict) -> Optional[Tuple[int, int]]:
    x, y, z = point_3d
    if z <= 0:
        return None
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    return (int(round(fx * x / z + cx)), int(round(fy * y / z + cy)))


def visualize_mask_trajectory(cam_dir: str) -> None:
    """Draw sam3_mask_traj on obs_frame -> obs_frame_sam3_traj.png."""
    cam_dir = Path(cam_dir)
    data = json.loads((cam_dir / "sam3_mask_traj.json").read_text())

    intrinsics = data["camera_info"]["intrinsics"]
    pixels = [
        px for pt in data["trajectory_3d"]
        if (px := _project_to_image(pt["position_3d"], intrinsics)) is not None
    ]

    img = cv2.imread(str(cam_dir / "obs_frame.png"))
    if img is None:
        print("  WARNING: cannot read obs_frame.png, skipping vis")
        return

    color = (0, 255, 0)
    for i in range(len(pixels) - 1):
        cv2.line(img, pixels[i], pixels[i + 1], color, 2)
    for px in pixels:
        cv2.circle(img, px, 3, color, -1)
    if pixels:
        cv2.circle(img, pixels[0], 5, (0, 0, 255), 2)   # start = red
        cv2.circle(img, pixels[-1], 5, (255, 0, 0), 2)  # end   = blue

    out_path = cam_dir / "obs_frame_sam3_traj.png"
    cv2.imwrite(str(out_path), img)
    print(f"  Saved vis: {out_path.name}  ({len(pixels)} points)")


# ---------------------------------------------------------------------------
# Pipeline discovery
# ---------------------------------------------------------------------------

def _discover_cam_dirs(
    dataset_roots: List[Path],
    skip_existing: bool,
    limit: Optional[int],
    existing: Optional[set] = None,
    shuffle: bool = False,
    seed: Optional[int] = 42,
) -> List[Path]:
    """Find cam dirs that have SAM3 tracking masks but haven't been processed yet."""
    cam_dirs: List[Path] = []
    for ep_dir in collect_episode_dirs(dataset_roots, limit, shuffle, seed):
        for iv_dir in sorted(ep_dir.iterdir()):
            if not (iv_dir.is_dir() and iv_dir.name.startswith("interval_")):
                continue
            for cam_dir in sorted(iv_dir.iterdir()):
                if not (cam_dir.is_dir() and cam_dir.name.startswith("cam_")):
                    continue
                if not ((cam_dir / "sam3_text_prompt_tracking_masks").is_dir()
                        and (cam_dir / "sam3_mask_meta.json").exists()
                        and (cam_dir / "trajectory.json").exists()):
                    continue
                key = f"{ep_dir.name}/{iv_dir.name}/{cam_dir.name}"
                if existing is not None and key in existing:
                    continue
                if skip_existing and (cam_dir / "sam3_mask_traj.json").exists():
                    continue
                cam_dirs.append(cam_dir)
                if limit and len(cam_dirs) >= limit:
                    return cam_dirs
    return cam_dirs


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------

def _process_cam_dir(
    cam_dir_str: str,
    output_root_str: str,
    min_3d: float,
    min_2d: float,
) -> Dict[str, Any]:
    """Compute + save trajectory for one cam dir. Returns a manifest status dict."""
    cam_dir = Path(cam_dir_str)
    try:
        result = compute_mask_trajectory(
            cam_dir_str, min_3d, min_2d, output_root=Path(output_root_str)
        )
        if result is None:
            return {"status": "filtered"}
        (cam_dir / "sam3_mask_traj.json").write_text(json.dumps(result, indent=2))
        visualize_mask_trajectory(cam_dir_str)
        return {"status": "done", "num_points": len(result["trajectory_3d"])}
    except Exception:
        return {"status": "error", "reason": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Pipeline entry point (called by mainprocess.py)
# ---------------------------------------------------------------------------

def run_sam3_mask_traj(cfg):
    """Stage 04: object trajectory from SAM3 tracking masks + sensor depth."""
    traj_cfg = cfg.sam3_mask_traj
    output_root = Path(cfg.output_root)
    dataset_roots = get_dataset_roots(cfg)
    skip_existing = cfg.skip_existing
    limit = cfg.get("limit")
    multi_processing = bool(traj_cfg.get("multi_processing", False))
    num_processes = max(1, int(traj_cfg.get("num_processes", 1)))
    min_3d = float(traj_cfg.min_start_end_dist_3d_m)
    min_2d = float(traj_cfg.min_start_end_dist_2d_px)
    flush_every = 10

    print("=" * 70)
    print("Stage 04 – SAM3 Mask Trajectory (mask centroid + depth)")
    print(f"  output_root     : {output_root}")
    print(f"  dataset dirs    : {[d.name for d in dataset_roots]}")
    print(f"  skip_existing   : {skip_existing}")
    print(f"  limit           : {limit}")
    print(f"  min_3d_dist     : {min_3d}")
    print(f"  min_2d_dist     : {min_2d}")
    print(f"  multi_processing: {multi_processing}  num_processes: {num_processes}")
    print("=" * 70)

    manifest = load_manifest(output_root, STAGE_SAM3_MASK_TRAJ) if skip_existing else {}
    existing = done_keys(manifest) if skip_existing else None

    shuffle = cfg.get("shuffle", False)
    seed = cfg.get("seed", 42)
    cam_dirs = _discover_cam_dirs(dataset_roots, skip_existing, limit, existing=existing, shuffle=shuffle, seed=seed)
    print(f"Cam directories to process: {len(cam_dirs)}")
    if not cam_dirs:
        print("Nothing to do.")
        return

    t0 = time.time()
    ok = filtered = fail = new_count = 0

    def _handle_entry(entry, cam_dir, idx, total):
        nonlocal ok, filtered, fail
        rel = cam_dir.relative_to(output_root)
        status = entry.get("status")
        if status == "done":
            ok += 1
            print(f"[{idx}/{total}] OK ({entry.get('num_points', '?')} pts)  {rel}")
        elif status == "filtered":
            filtered += 1
            print(f"[{idx}/{total}] filtered  {rel}")
        else:
            fail += 1
            print(f"[{idx}/{total}] ERROR  {rel}: {entry.get('reason', '')}")

    if multi_processing:
        import ray

        started_ray = False
        total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=num_processes,
                     num_gpus=total_gpus)
            started_ray = True

        try:
            # SVO reader (ZED SDK) needs GPU access — give each worker a small fraction
            gpus_per_worker = total_gpus / max(num_processes, 1) if total_gpus > 0 else 0
            process_remote = ray.remote(num_cpus=1, num_gpus=gpus_per_worker)(_process_cam_dir)
            pending: Dict[Any, Tuple[Path, str]] = {}
            for cam_dir in cam_dirs:
                key = f"{cam_dir.parent.parent.name}/{cam_dir.parent.name}/{cam_dir.name}"
                ref = process_remote.remote(str(cam_dir), str(output_root), min_3d, min_2d)
                pending[ref] = (cam_dir, key)

            total = len(pending)
            idx = 0
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                ref = ready[0]
                cam_dir, key = pending.pop(ref)
                idx += 1
                try:
                    entry = ray.get(ref)
                except Exception:
                    entry = {"status": "error", "reason": traceback.format_exc()}
                manifest[key] = entry
                new_count += 1
                _handle_entry(entry, cam_dir, idx, total)
                if new_count % flush_every == 0:
                    save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)
        finally:
            save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)
            if started_ray:
                ray.shutdown()

    else:
        total = len(cam_dirs)
        for idx, cam_dir in enumerate(cam_dirs, 1):
            key = f"{cam_dir.parent.parent.name}/{cam_dir.parent.name}/{cam_dir.name}"
            entry = _process_cam_dir(str(cam_dir), str(output_root), min_3d, min_2d)
            manifest[key] = entry
            new_count += 1
            _handle_entry(entry, cam_dir, idx, total)
            if new_count % flush_every == 0:
                save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)
        save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done: {ok} succeeded, {filtered} filtered, {fail} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
