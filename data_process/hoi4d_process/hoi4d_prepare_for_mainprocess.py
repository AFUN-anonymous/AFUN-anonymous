#!/usr/bin/env python3
"""
Prepare the HOI4D preprocess output for mainprocess stages (query_gen_vllm + sam3_mask).

Creates:
  1. episode_summary.json per sequence (required by mainprocess discovery)
  2. Symlinks mid_frame.png -> contact_frame.png (VLM expects contact_frame)
  3. Renames sam_mask.png -> whole_object_mask.png (preserve before SAM3 overwrites)

Usage:
    python -m hoi4d_process.hoi4d_prepare_for_mainprocess
    python -m hoi4d_process.hoi4d_prepare_for_mainprocess --dry-run
"""

import argparse
import json
from pathlib import Path

from .config import OUTPUT_ROOT


def prepare(output_root: Path, dry_run: bool = False):
    # Walk all sequence dirs (hoi4d_H1_C13_N16_...)
    seq_dirs = sorted(d for d in output_root.iterdir()
                      if d.is_dir() and d.name.startswith("hoi4d_"))

    print(f"Sequences: {len(seq_dirs)}")
    total_summaries = 0
    total_symlinks = 0
    total_renames = 0

    for seq_dir in seq_dirs:
        # Count intervals
        interval_dirs = sorted(d for d in seq_dir.iterdir()
                               if d.is_dir() and d.name.startswith("interval_"))

        if not interval_dirs:
            continue

        # 1. Create episode_summary.json
        summary_path = seq_dir / "episode_summary.json"
        if not summary_path.exists():
            first_meta_path = interval_dirs[0] / "cam_0" / "metadata.json"
            if not first_meta_path.exists():
                continue

            summary = {
                "episode_id": seq_dir.name,
                "episode_path": str(seq_dir),
                "num_frames": 300,
                "num_intervals": len(interval_dirs),
                "num_cams": 1,
                "fps": 15,
                "intervals": [],
            }
            for iv_dir in interval_dirs:
                iv_meta_path = iv_dir / "cam_0" / "metadata.json"
                if iv_meta_path.exists():
                    with open(iv_meta_path) as f:
                        iv_meta = json.load(f)
                    summary["intervals"].append({
                        "idx": iv_meta.get("interval_idx", 0),
                        "start": iv_meta.get("obs_frame", 0),
                        "end": iv_meta.get("end_frame", 299),
                        "length": iv_meta.get("end_frame", 299) - iv_meta.get("obs_frame", 0) + 1,
                    })

            if not dry_run:
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
            total_summaries += 1

        # 2 & 3. Process each interval's cam_0
        for iv_dir in interval_dirs:
            cam_dir = iv_dir / "cam_0"
            if not cam_dir.exists():
                continue

            # Symlink mid_frame.png -> contact_frame.png
            mid = cam_dir / "mid_frame.png"
            contact = cam_dir / "contact_frame.png"
            if mid.exists() and not contact.exists():
                if not dry_run:
                    contact.symlink_to(mid.name)
                total_symlinks += 1

            # Rename sam_mask.png -> whole_object_mask.png
            sam_mask = cam_dir / "sam_mask.png"
            whole_mask = cam_dir / "whole_object_mask.png"
            if sam_mask.exists() and not whole_mask.exists():
                if not dry_run:
                    sam_mask.rename(whole_mask)
                total_renames += 1

    print(f"{'[DRY RUN] ' if dry_run else ''}Done:")
    print(f"  episode_summary.json created: {total_summaries}")
    print(f"  contact_frame.png symlinks:   {total_symlinks}")
    print(f"  sam_mask.png -> whole_object:  {total_renames}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    prepare(Path(args.output_root), args.dry_run)


if __name__ == "__main__":
    main()
