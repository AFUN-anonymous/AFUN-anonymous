#!/usr/bin/env python3
"""
04_sam3_mask_traj_depth.py - Object trajectory from SAM3 masks + DA3 video depth.

Variant of ``04_sam3_mask_traj.py`` for datasets without usable sensor depth
(agibot, robomind2, vitra_*): depth comes from DA3-Streaming video prediction
(``preprocess.depth_predictor.DepthPredictor.predict_video``).

Per cam directory (listed from the stage 02.5 manifest):
  1. Sample frames obs_frame, obs_frame + depth_stride, ... up to end_frame that
     have a tracked mask, plus obs_frame itself.
  2. Load their RGB with the dataset preprocessor and run DA3-Streaming once on
     the sampled frames (single chunk) to get metric depth and per-frame
     intrinsics.
  3. For each sampled mask frame, back-project every mask pixel with valid depth
     using DA3's intrinsics and take the component-wise median 3D point.
  4. Transform it into the obs_frame camera with
     ``preprocessor.reproject_camera_point_to_obs_frame`` (identity for static
     cameras, SLAM poses for VITRA) and project to obs_frame pixels.
  5. Filter near-static tracks (min_start_end_dist_3d_m / _2d_px) and save
     sam3_mask_traj.json (intrinsics = DA3 obs-frame intrinsics), the overlay
     image, and obs_frame_depth.npy (DA3 depth of obs_frame, mm).

Run:
  python mainprocess.py stage=sam3_mask_traj_depth dataset_name=vitra_epic skip_existing=true
  # parallel shards: sam3_mask_traj_depth.num_shards=N sam3_mask_traj_depth.shard_idx=i
"""

import json
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import (
    STAGE_SAM3_MASK_TRAJ,
    STAGE_SAM3_VIDEO_TRACK,
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
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 0


def _mask_to_3d(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: Dict[str, Any],
) -> Optional[Tuple[List[float], float, float]]:
    """Back-project mask pixels → component-wise median 3D point (cam frame, meters)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    depths = depth_mm[ys, xs].astype(np.float64)
    valid = depths > 0
    if not np.any(valid):
        return None

    xv = xs[valid].astype(np.float64)
    yv = ys[valid].astype(np.float64)
    dv = depths[valid]

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    z = dv / 1000.0  # mm → m
    x3d = (xv - cx) * z / fx
    y3d = (yv - cy) * z / fy

    pos_3d = [float(np.median(x3d)), float(np.median(y3d)), float(np.median(z))]
    if pos_3d[2] > 0:
        cx_px = fx * pos_3d[0] / pos_3d[2] + cx
        cy_px = fy * pos_3d[1] / pos_3d[2] + cy
    else:
        cx_px, cy_px = float(np.median(xv)), float(np.median(yv))
    return pos_3d, float(cx_px), float(cy_px)


def _start_end_distance(points: List[Dict[str, Any]], key: str) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(
        np.asarray(points[-1][key], dtype=np.float32)
        - np.asarray(points[0][key], dtype=np.float32)
    ))


# ---------------------------------------------------------------------------
# Core: compute trajectory using video depth
# ---------------------------------------------------------------------------

def compute_mask_trajectory_with_video_depth(
    cam_dir: str,
    cfg,
    min_start_end_dist_3d_m: float,
    min_start_end_dist_2d_px: float,
    depth_stride: int,
    output_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute the object trajectory with DA3-Streaming video depth.

    Depth is predicted once per interval on every depth_stride-th frame;
    trajectory points exist only at those frames (no interpolation).
    Returns the sam3_mask_traj.json dict, or None if filtered.
    """
    cam_dir = Path(cam_dir)
    mask_meta = json.loads((cam_dir / "sam3_mask_meta.json").read_text())
    stride_mask: int = mask_meta["stride"]
    interval_meta = json.loads((cam_dir.parent / "metadata.json").read_text())
    obs_frame_idx: int = interval_meta.get("obs_frame", interval_meta["start_frame"]) or 0
    interval_end: int = interval_meta["end_frame"]
    # frame_00000 = obs_frame (see stage 02.5).
    mask_start_frame: int = interval_meta.get("obs_frame", 0) or 0

    traj_data = json.loads((cam_dir / "trajectory.json").read_text())
    camera_info = traj_data["camera_info"]
    # Dataset intrinsics are kept for provenance only; back-projection uses
    # DA3's predicted intrinsics.
    dataset_intrinsics = camera_info["intrinsics"]
    cam_serial = camera_info["cam_id"]

    episode_summary = json.loads(
        (cam_dir.parent.parent / "episode_summary.json").read_text()
    )
    episode_path = episode_summary["episode_path"]

    mask_files = sorted((cam_dir / "sam3_text_prompt_tracking_masks").glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files in {cam_dir}/sam3_text_prompt_tracking_masks")

    # Map: video_frame_idx → mask path
    mask_by_frame: Dict[int, Path] = {}
    for mask_path in mask_files:
        local_idx = int(mask_path.stem.split("_")[-1])
        video_frame_idx = mask_start_frame + local_idx * stride_mask
        mask_by_frame[video_frame_idx] = mask_path

    # Depth frames: stride-sampled frames that have a mask, plus obs_frame
    # (always, for obs_frame_depth.npy, even if tracking missed it).
    traj_sampled = list(range(obs_frame_idx, interval_end + 1, depth_stride))
    traj_sampled = [f for f in traj_sampled if f in mask_by_frame]

    # Include obs_frame unconditionally (for obs_frame_depth.npy)
    sampled_set = set(traj_sampled) | {obs_frame_idx}
    sampled_frames = sorted(sampled_set)

    if len(traj_sampled) < 2:
        # Graceful filter: not enough mask frames to produce a trajectory
        print(
            f"  Skipping (filtered): only {len(traj_sampled)} sampled mask frames. "
            f"interval=[{obs_frame_idx}, {interval_end}], "
            f"stride={depth_stride}, mask_frames={len(mask_by_frame)}"
        )
        return None

    rel = cam_dir.relative_to(output_root) if output_root else cam_dir
    print(f"Cam: {rel}")
    print(f"  interval [{obs_frame_idx},{interval_end}] "
          f"({interval_end - obs_frame_idx + 1} frames), "
          f"masks={len(mask_by_frame)}, sampled={len(sampled_frames)} "
          f"(stride={depth_stride})")

    dataset_name = cam_dir.parent.parent.parent.name
    if output_root is None:
        output_root = cam_dir.parent.parent.parent.parent
    preprocessor = get_dataset_preprocessor(dataset_name, output_root)
    episode_data = preprocessor.load_episode(episode_path)

    trajectory_3d: List[Dict] = []
    trajectory_2d: List[Dict] = []
    skipped = 0

    try:
        # ----- Step 1: load sampled RGB frames -----
        # DA3-Streaming writes frames via cv2.imwrite which expects BGR.
        # preprocessor.extract_frame returns RGB; convert to BGR for the dict.
        rgb_frames: Dict[int, np.ndarray] = {}
        for fi in sampled_frames:
            rgb = preprocessor.extract_frame(episode_data, cam_serial, fi)
            if rgb is None:
                continue
            rgb_frames[fi] = rgb[..., ::-1]  # RGB → BGR

        if len(rgb_frames) < 2:
            raise RuntimeError(f"Not enough RGB frames loaded: {len(rgb_frames)}")
        print(f"  Loaded {len(rgb_frames)} RGB frames")

        # ----- Step 2: run video depth in a temp work dir (auto-cleaned) -----
        from omegaconf import OmegaConf
        from preprocess.depth_predictor import DepthPredictor

        n = len(rgb_frames)
        # Force a single chunk (chunk_size > n) with overlap 0: with a single
        # chunk and overlap > 0, DA3-Streaming leaves the trailing overlap
        # poses of the first chunk as None and crashes in save_camera_poses.
        configured_chunk = int(cfg.depth_prediction.video.chunk_size)
        effective_chunk = max(n + 1, configured_chunk)
        cfg_for_video = OmegaConf.create({
            "depth_prediction": {
                "video": {
                    "model": cfg.depth_prediction.video.model,
                    "chunk_size": effective_chunk,
                    "overlap": 0,
                }
            }
        })

        with tempfile.TemporaryDirectory(prefix="da3_video_") as work_dir_str:
            work_dir = Path(work_dir_str)
            t0 = time.time()
            depth_maps, intrinsics_by_frame = DepthPredictor.predict_video(
                rgb_frames, work_dir, cfg_for_video
            )
            print(f"  Video depth: {len(depth_maps)} maps in {time.time()-t0:.1f}s")

            if obs_frame_idx not in depth_maps:
                raise RuntimeError(
                    f"obs_frame {obs_frame_idx} missing from video depth. "
                    f"Got frames: {sorted(depth_maps.keys())[:5]}..."
                )
            obs_intrinsics = intrinsics_by_frame.get(obs_frame_idx)
            if obs_intrinsics is None:
                raise RuntimeError(
                    f"DA3 intrinsics missing for obs_frame {obs_frame_idx}"
                )

            # ----- Step 3: per sampled frame, back-project + reproject -----
            # Iterate only frames with masks (traj_sampled), not the full
            # sampled_frames which may include obs_frame without a mask.
            # Back-projection uses DA3's per-frame predicted intrinsics (not
            # the dataset-provided intrinsics) — DA3's depth and intrinsics
            # are internally consistent; mixing dataset intrinsics with DA3
            # depth introduces a scale/principal-point mismatch.
            for fi in traj_sampled:
                mask = _load_mask(mask_by_frame[fi])
                if not np.any(mask):
                    skipped += 1
                    continue

                depth_mm = depth_maps.get(fi)
                frame_intrinsics = intrinsics_by_frame.get(fi)
                if depth_mm is None or frame_intrinsics is None:
                    skipped += 1
                    continue

                result = _mask_to_3d(mask, depth_mm.astype(np.float32), frame_intrinsics)
                if result is None:
                    skipped += 1
                    continue

                pos_3d, cx_px, cy_px = result
                pos_3d_np = np.asarray(pos_3d, dtype=np.float64)

                # Reproject to obs_frame coords (identity for static cams;
                # T_w2c_obs @ inv(T_w2c_t) for VITRA/moving cameras).
                pos_3d_obs = preprocessor.reproject_camera_point_to_obs_frame(
                    episode_data, pos_3d_np, fi, obs_frame_idx
                )
                if not np.allclose(pos_3d_obs, pos_3d_np, atol=1e-8):
                    z_obs = float(pos_3d_obs[2])
                    if z_obs <= 0:
                        skipped += 1
                        continue
                    fx, fy = obs_intrinsics["fx"], obs_intrinsics["fy"]
                    cx_cam, cy_cam = obs_intrinsics["cx"], obs_intrinsics["cy"]
                    cx_px = fx * float(pos_3d_obs[0]) / z_obs + cx_cam
                    cy_px = fy * float(pos_3d_obs[1]) / z_obs + cy_cam

                trajectory_3d.append({
                    "frame_idx": fi,
                    "position_3d": [float(x) for x in pos_3d_obs],
                })
                trajectory_2d.append({
                    "frame_idx": fi,
                    "position_pixel": [round(cx_px, 2), round(cy_px, 2)],
                })

            print(f"  Trajectory: {len(trajectory_3d)} points, {skipped} skipped")

            # ----- Step 4: save obs_frame_depth.npy (overwrite) -----
            # Only keep the obs_frame depth as a permanent artifact. The
            # per-frame depth in depth_maps + DA3-Streaming tmp dir are
            # discarded on TemporaryDirectory context exit.
            obs_depth_mm = depth_maps[obs_frame_idx].astype(np.float32)
            np.save(str(cam_dir / "obs_frame_depth.npy"), obs_depth_mm)
            print(f"  Saved obs_frame_depth.npy ({obs_depth_mm.shape}, "
                  f"{obs_depth_mm.min():.0f}-{obs_depth_mm.max():.0f} mm)")
        # <-- work_dir auto-cleaned here

    finally:
        preprocessor.close_episode(episode_data)

    # ----- Step 5: filter near-static trajectories -----
    dist_3d = _start_end_distance(trajectory_3d, "position_3d")
    dist_2d = _start_end_distance(trajectory_2d, "position_pixel")
    if dist_3d < min_start_end_dist_3d_m or dist_2d < min_start_end_dist_2d_px:
        print(
            f"  Filtering near-static: 3D={dist_3d:.4f}m < {min_start_end_dist_3d_m:.4f}m "
            f"or 2D={dist_2d:.2f}px < {min_start_end_dist_2d_px:.2f}px"
        )
        return None

    # Overwrite intrinsics with DA3's obs-frame intrinsics: the 3D trajectory
    # was back-projected using DA3 intrinsics, so downstream 3D→2D projection
    # (visualize_mask_trajectory, curve-fit convert_cc_to_fsframe) must use
    # the same intrinsics to stay consistent. Preserve the dataset-provided
    # intrinsics under a separate key for provenance.
    camera_info_out = dict(camera_info)
    camera_info_out["intrinsics"] = obs_intrinsics
    camera_info_out["intrinsics_dataset"] = dataset_intrinsics
    camera_info_out["intrinsics_source"] = "da3_video_obs_frame"

    return {
        "camera_info": camera_info_out,
        "trajectory_3d": trajectory_3d,
        "trajectory_2d": trajectory_2d,
        "depth_stride": depth_stride,
    }


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
    """Draw sam3_mask_traj on obs_frame → obs_frame_sam3_traj.png."""
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
    output_root: Optional[Path] = None,
) -> List[Path]:
    """Drive discovery from the upstream tracking manifest.

    Instead of walking every episode dir on disk, we iterate ``done`` keys
    from ``STAGE_SAM3_VIDEO_TRACK`` (tracking's output) and stat only those
    cam dirs. Per-iteration cost is O(#tracked intervals), not O(all
    episodes on disk).
    """
    cam_dirs: List[Path] = []
    if output_root is None:
        return cam_dirs

    track_manifest = load_manifest(output_root, STAGE_SAM3_VIDEO_TRACK)
    if not track_manifest:
        return cam_dirs

    track_keys = [
        k for k, v in track_manifest.items()
        if isinstance(v, dict) and v.get("status") != "error"
    ]
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(track_keys)
    else:
        track_keys.sort()

    for key in track_keys:
        parts = key.split("/")
        if len(parts) != 3:
            continue
        ep_id, iv_name, cam_name = parts

        if existing is not None and key in existing:
            continue

        cam_dir: Optional[Path] = None
        for root in dataset_roots:
            candidate = root / ep_id / iv_name / cam_name
            if candidate.is_dir():
                cam_dir = candidate
                break
        if cam_dir is None:
            continue

        if not ((cam_dir / "sam3_text_prompt_tracking_masks").is_dir()
                and (cam_dir / "sam3_mask_meta.json").exists()
                and (cam_dir / "trajectory.json").exists()):
            continue
        if skip_existing and (cam_dir / "sam3_mask_traj.json").exists():
            continue

        cam_dirs.append(cam_dir)
        if limit and len(cam_dirs) >= limit:
            return cam_dirs
    return cam_dirs


# ---------------------------------------------------------------------------
# Worker (one cam dir)
# ---------------------------------------------------------------------------

def _process_cam_dir(
    cam_dir_str: str,
    output_root_str: str,
    cfg,
    min_3d: float,
    min_2d: float,
    depth_stride: int,
) -> Dict[str, Any]:
    cam_dir = Path(cam_dir_str)
    try:
        result = compute_mask_trajectory_with_video_depth(
            cam_dir_str, cfg, min_3d, min_2d, depth_stride,
            output_root=Path(output_root_str),
        )
        if result is None:
            return {"status": "filtered"}
        (cam_dir / "sam3_mask_traj.json").write_text(json.dumps(result, indent=2))
        visualize_mask_trajectory(cam_dir_str)
        return {"status": "done", "num_points": len(result["trajectory_3d"])}
    except Exception:
        return {"status": "error", "reason": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_sam3_mask_traj_depth(cfg):
    """Stage 04 (video depth): object trajectory from SAM3 masks + DA3-Streaming depth."""
    depth_cfg = cfg.sam3_mask_traj_depth
    depth_stride = int(depth_cfg.depth_stride)

    output_root = Path(cfg.output_root)
    dataset_roots = get_dataset_roots(cfg)
    skip_existing = cfg.skip_existing
    limit = cfg.get("limit")
    min_3d = float(depth_cfg.min_start_end_dist_3d_m)
    min_2d = float(depth_cfg.min_start_end_dist_2d_px)

    flush_every = 5

    print("=" * 70)
    print("Stage 04 – SAM3 Mask Trajectory (DA3-Streaming video depth)")
    print(f"  output_root     : {output_root}")
    print(f"  dataset dirs    : {[d.name for d in dataset_roots]}")
    print(f"  depth_stride    : {depth_stride}  "
          f"(e.g. 30fps → {30/depth_stride:.1f}fps)")
    print(f"  skip_existing   : {skip_existing}")
    print(f"  limit           : {limit}")
    print(f"  min_3d_dist     : {min_3d}")
    print(f"  min_2d_dist     : {min_2d}")
    print("=" * 70)

    # The on-disk manifest only defines the skip-set; this run's entries go in
    # a fresh dict so each save pushes only our diff (several shards may save
    # to the same manifest concurrently).
    _existing_manifest = load_manifest(output_root, STAGE_SAM3_MASK_TRAJ) if skip_existing else {}
    existing = done_keys(_existing_manifest) if skip_existing else None
    manifest: Dict[str, Any] = {}

    shuffle = cfg.get("shuffle", False)
    seed = cfg.get("seed", 42)
    cam_dirs = _discover_cam_dirs(
        dataset_roots, skip_existing, limit,
        existing=existing, shuffle=shuffle, seed=seed,
        output_root=output_root,
    )

    num_shards = int(depth_cfg.num_shards)
    shard_idx = int(depth_cfg.shard_idx)
    if num_shards < 1 or not (0 <= shard_idx < num_shards):
        raise ValueError(
            f"Invalid sharding: num_shards={num_shards}, shard_idx={shard_idx}"
        )
    if num_shards > 1:
        total_before = len(cam_dirs)
        cam_dirs = [d for i, d in enumerate(cam_dirs) if i % num_shards == shard_idx]
        print(f"Shard {shard_idx}/{num_shards}: {len(cam_dirs)} / {total_before} cam_dirs")

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
            reason = entry.get("reason", "")
            # Show only the last (exception) line of the traceback
            reason_line = reason.strip().split("\n")[-1] if reason else ""
            print(f"[{idx}/{total}] ERROR  {rel}: {reason_line}")

    # Cam dirs processed concurrently; each worker process loads its own DA3 model.
    num_workers = int(depth_cfg.num_workers)
    total = len(cam_dirs)

    if num_workers <= 1:
        for idx, cam_dir in enumerate(cam_dirs, 1):
            key = f"{cam_dir.parent.parent.name}/{cam_dir.parent.name}/{cam_dir.name}"
            entry = _process_cam_dir(
                str(cam_dir), str(output_root), cfg, min_3d, min_2d, depth_stride,
            )
            manifest[key] = entry
            new_count += 1
            _handle_entry(entry, cam_dir, idx, total)
            if new_count % flush_every == 0:
                save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)
    else:
        # spawn, not fork: the parent may already hold a CUDA context. Each
        # child loads its own DA3 model on first use and reuses it.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"Running with num_workers={num_workers} (parallel)")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _process_cam_dir,
                    str(cam_dir),
                    str(output_root),
                    cfg,
                    min_3d,
                    min_2d,
                    depth_stride,
                ): cam_dir
                for cam_dir in cam_dirs
            }
            completed = 0
            for fut in as_completed(futures):
                cam_dir = futures[fut]
                completed += 1
                key = f"{cam_dir.parent.parent.name}/{cam_dir.parent.name}/{cam_dir.name}"
                try:
                    entry = fut.result()
                except Exception:
                    entry = {"status": "error", "reason": traceback.format_exc()}
                manifest[key] = entry
                new_count += 1
                _handle_entry(entry, cam_dir, completed, total)
                if new_count % flush_every == 0:
                    save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)

    save_manifest(output_root, STAGE_SAM3_MASK_TRAJ, manifest)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done: {ok} succeeded, {filtered} filtered, {fail} failed  ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
