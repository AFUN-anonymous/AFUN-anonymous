#!/usr/bin/env python3
"""
HOI4D preprocessing (custom path; no preprocess.py adapter).

For each sequence in release.txt with action annotations:
  - intervals from the action events (Reachout ... next rest/stop)
  - whole-object mask from 2Dseg at obs_frame
  - object 3D trajectory in the obs_frame camera: 2Dseg mask centroid,
    back-projected with objpose depth, moved through the camera extrinsics
  - task language from task_definitions.csv
Intervals whose object moves < MIN_WORLD_DISPLACEMENT_M are dropped.
Paths are set by environment variables (see hoi4d_process/config.py).

Usage:
    python -m hoi4d_process.process_hoi4d [--skip-existing] [--limit N] [--category C13]
"""

import argparse
import json
import time
import traceback
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from .config import (
    RGB_ROOT, ANNO_ROOT, DEPTH_ROOT, CAMERA_ROOT, OUTPUT_ROOT, RELEASE_TXT,
    CATEGORIES, MIN_WORLD_DISPLACEMENT_M,
)
from .hoi4d_utils import (
    parse_action_events, detect_intervals,
    load_intrinsics, load_camera_extrinsics,
    extract_mask, extract_trajectory,
    decode_frame, decode_depth_frame,
    visualize_trajectory, make_language, get_task_description,
)


def parse_sequence_path(seq_path: str) -> dict:
    """Parse 'ZY20210800001/H1/C13/N16/S132/s04/T1' into components."""
    parts = seq_path.strip("/").split("/")
    return {
        "device_id": parts[0],
        "human_id": parts[1],
        "category_id": parts[2],
        "instance_id": parts[3],
        "room_id": parts[4],
        "layout_id": parts[5],
        "task_id": parts[6],
    }


def make_sequence_id(seq_path: str) -> str:
    """'ZY.../H1/C13/N16/S132/s04/T1' -> 'hoi4d_H1_C13_N16_S132_s04_T1'"""
    p = parse_sequence_path(seq_path)
    return f"hoi4d_{p['human_id']}_{p['category_id']}_{p['instance_id']}_{p['room_id']}_{p['layout_id']}_{p['task_id']}"


def process_sequence(
    seq_path: str,
    anno_root: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> dict:
    """Process one HOI4D sequence end-to-end."""

    info = parse_sequence_path(seq_path)
    seq_id = make_sequence_id(seq_path)
    category_id = info["category_id"]
    task_id = info["task_id"]
    device_id = info["device_id"]
    category_name = CATEGORIES.get(category_id, "unknown")

    # --- Check data availability ---
    seq_anno = anno_root / seq_path
    seq_rgb = RGB_ROOT / seq_path / "align_rgb" / "image.mp4"
    action_json = seq_anno / "action" / "color.json"
    objpose_dir = seq_anno / "objpose"
    mask_dir = seq_anno / "2Dseg" / "mask"
    extrinsics_log = seq_anno / "3Dseg" / "output.log"
    intrinsics_path = CAMERA_ROOT / device_id / "intrin.npy"

    if not action_json.exists():
        return {"status": "skip", "reason": "no_action_json"}
    if not seq_rgb.exists():
        return {"status": "skip", "reason": "no_rgb_video"}
    if not objpose_dir.exists():
        return {"status": "skip", "reason": "no_objpose"}
    if not mask_dir.exists():
        return {"status": "skip", "reason": "no_2dseg"}
    if not intrinsics_path.exists():
        return {"status": "skip", "reason": "no_intrinsics"}

    # --- Load shared data ---
    intrinsics = load_intrinsics(intrinsics_path)

    extrinsics = []
    if extrinsics_log.exists():
        extrinsics = load_camera_extrinsics(extrinsics_log)

    action_events = parse_action_events(action_json)
    intervals = detect_intervals(action_events)

    if not intervals:
        return {"status": "skip", "reason": "no_intervals"}

    # --- Open RGB video ---
    rgb_cap = cv2.VideoCapture(str(seq_rgb))
    if not rgb_cap.isOpened():
        return {"status": "error", "reason": "cannot_open_rgb"}

    # --- Optional: open depth video ---
    depth_cap = None
    seq_depth = DEPTH_ROOT / seq_path / "align_depth" / "depth_video.avi"
    if seq_depth.exists():
        depth_cap = cv2.VideoCapture(str(seq_depth))
        if not depth_cap.isOpened():
            depth_cap = None

    # --- Process each interval ---
    processed = 0
    skipped = 0
    filtered = 0

    for iv_idx, interval in enumerate(intervals):
        out_dir = output_root / seq_id / f"interval_{iv_idx:03d}" / "cam_0"

        # Skip existing
        if skip_existing and (out_dir / "sam3_mask_traj.json").exists():
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Decode frames ---
        obs_rgb = decode_frame(rgb_cap, interval.obs_frame)
        mid_rgb = decode_frame(rgb_cap, interval.mid_frame)
        if obs_rgb is None:
            continue

        # --- Extract mask ---
        mask = extract_mask(mask_dir, interval.obs_frame)
        if mask is None:
            skipped += 1
            continue

        # --- Extract trajectory ---
        if not extrinsics:
            skipped += 1
            continue

        traj_points, max_disp = extract_trajectory(
            objpose_dir, extrinsics, intrinsics,
            interval.obs_frame, interval.end_frame,
            mask_dir=mask_dir,
        )

        if len(traj_points) < 2:
            skipped += 1
            continue

        # --- Motion filter ---
        if max_disp < MIN_WORLD_DISPLACEMENT_M:
            filtered += 1
            continue

        # --- Save outputs ---

        # obs_frame.png
        Image.fromarray(obs_rgb).save(str(out_dir / "obs_frame.png"))

        # mid_frame.png
        if mid_rgb is not None:
            Image.fromarray(mid_rgb).save(str(out_dir / "mid_frame.png"))

        # sam_mask.png
        cv2.imwrite(str(out_dir / "sam_mask.png"), mask)

        # obs_frame_depth.npy (if depth available)
        depth_source = "none"
        if depth_cap is not None:
            depth_frame = decode_depth_frame(depth_cap, interval.obs_frame)
            if depth_frame is not None:
                np.save(str(out_dir / "obs_frame_depth.npy"), depth_frame)
                depth_source = "dataset"

        # sam3_mask_traj.json
        traj_json = {
            "camera_info": {
                "intrinsics": intrinsics,
                "cam_id": device_id,
            },
            "trajectory_3d": [
                {"frame_idx": p.frame_idx, "position_3d": p.position_3d}
                for p in traj_points
            ],
            "trajectory_2d": [
                {"frame_idx": p.frame_idx, "position_pixel": p.position_pixel}
                for p in traj_points if p.position_pixel is not None
            ],
        }
        with open(out_dir / "sam3_mask_traj.json", "w") as f:
            json.dump(traj_json, f, indent=2)

        # trajectory.json (camera_info for downstream compatibility)
        with open(out_dir / "trajectory.json", "w") as f:
            json.dump({"camera_info": traj_json["camera_info"]}, f, indent=2)

        # metadata.json
        task_def = get_task_description(category_id, task_id)
        language = make_language(category_id, task_id)
        metadata = {
            "dataset": "hoi4d",
            "sequence_path": seq_path,
            "sequence_id": seq_id,
            "interval_idx": iv_idx,
            "obs_frame": interval.obs_frame,
            "mid_frame": interval.mid_frame,
            "end_frame": interval.end_frame,
            "category_id": category_id,
            "category_name": category_name,
            "task_id": task_id,
            "task_definition": task_def,
            "interval_language": language,
            "sam3_prompt": category_name.lower(),
            "max_world_displacement_m": round(max_disp, 4),
            "num_trajectory_points": len(traj_points),
            "num_cams": 1,
            "action_events": [
                {"event": e.event, "start": e.start_frame, "end": e.end_frame}
                for e in interval.events
            ],
            "cam_0": {
                "cam_id": device_id,
                "intrinsic_source": "calibration",
                "depth_source": depth_source,
            },
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Visualization
        visualize_trajectory(obs_rgb, traj_points, out_dir / "obs_frame_sam3_traj.png")

        processed += 1

    rgb_cap.release()
    if depth_cap is not None:
        depth_cap.release()

    return {
        "status": "done",
        "sequence_id": seq_id,
        "intervals": len(intervals),
        "processed": processed,
        "skipped": skipped,
        "filtered": filtered,
    }


def main():
    parser = argparse.ArgumentParser(description="HOI4D preprocessing")
    parser.add_argument("--limit", type=int, default=None, help="Max sequences to process")
    parser.add_argument("--category", type=str, default=None, help="Filter by category (e.g., C13)")
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    anno_root = ANNO_ROOT

    # Load sequence list
    with open(RELEASE_TXT) as f:
        all_sequences = [line.strip() for line in f if line.strip()]

    # Filter by category
    if args.category:
        all_sequences = [s for s in all_sequences if f"/{args.category}/" in s]

    # Filter by available annotations
    available = []
    for seq in all_sequences:
        if (anno_root / seq / "action" / "color.json").exists():
            available.append(seq)

    if args.limit:
        available = available[:args.limit]

    print("=" * 70)
    print("HOI4D preprocessing")
    print(f"  anno_root    : {anno_root}")
    print(f"  output_root  : {output_root}")
    print(f"  sequences    : {len(available)} / {len(all_sequences)} (with annotations)")
    if args.category:
        print(f"  category     : {args.category} ({CATEGORIES.get(args.category, '?')})")
    print(f"  skip_existing: {args.skip_existing}")
    print("=" * 70)

    t0 = time.time()
    total_ok, total_skip, total_filter, total_err = 0, 0, 0, 0

    for idx, seq in enumerate(available, 1):
        print(f"\n[{idx}/{len(available)}] {seq}")
        try:
            result = process_sequence(seq, anno_root, output_root, args.skip_existing)
            status = result["status"]
            if status == "done":
                print(f"  -> {result['processed']} processed, "
                      f"{result['skipped']} skipped, "
                      f"{result['filtered']} filtered (motion<{MIN_WORLD_DISPLACEMENT_M}m)")
                total_ok += result["processed"]
                total_skip += result["skipped"]
                total_filter += result["filtered"]
            elif status == "skip":
                print(f"  -> SKIP: {result['reason']}")
                total_skip += 1
            else:
                print(f"  -> ERROR: {result.get('reason', '?')}")
                total_err += 1
        except Exception as e:
            print(f"  -> ERROR: {e}")
            traceback.print_exc()
            total_err += 1

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Processed: {total_ok}")
    print(f"  Skipped:   {total_skip}")
    print(f"  Filtered:  {total_filter} (world displacement < {MIN_WORLD_DISPLACEMENT_M}m)")
    print(f"  Errors:    {total_err}")
    print(f"  Output:    {output_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
