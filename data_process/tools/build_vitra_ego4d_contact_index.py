"""
Build the VITRA Ego4D episode -> contact frame index (config: vitra_ego4d.yaml,
ego4d_contact_index; also excluded from the vitra_ego4d_noannot episode list).

For each VITRA episode .npy file, find the best-matching Ego4D narrated action
(same video_uid, maximum frame overlap) and record the contact_frame mapped to
the local frame index within the VITRA episode.

Usage:
    python tools/build_vitra_ego4d_contact_index.py \
        --fho-main /path/to/Ego4D/v2/annotations/fho_main.json \
        --vitra-root /path/to/VITRA-1M \
        --output /path/to/VITRA-1M/ego4d_contact_index_full.json

Output JSON format:
    {
        "<episode_stem>": {
            "contact_frame_local": int,   # index into video_decode_frame array
            "contact_frame_ego4d": int,   # absolute frame in the Ego4D video
            "narration": str,
            "ego4d_start": int,
            "ego4d_end": int,
            "overlap_ratio": float
        },
        ...
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build VITRA -> Ego4D contact frame index")
    parser.add_argument(
        "--fho-main",
        required=True,
        help="Path to Ego4D fho_main.json",
    )
    parser.add_argument(
        "--vitra-root",
        required=True,
        help="Root directory of VITRA-1M dataset",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["ego4d_cooking_and_cleaning", "ego4d_other"],
        help="VITRA subset directories to process",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.3,
        help="Minimum overlap ratio to include an episode (default: 0.3)",
    )
    return parser.parse_args()


def load_fho_main(fho_main_path: str) -> dict[str, list[dict]]:
    """
    Load fho_main.json and build an index:
        video_uid -> list of valid actions with contact_frame

    Each action dict contains:
        start_frame, end_frame, contact_frame, narration_text
    """
    print(f"Loading {fho_main_path} ...")
    with open(fho_main_path) as f:
        fho = json.load(f)

    video_actions: dict[str, list[dict]] = defaultdict(list)
    total_actions = 0
    kept_actions = 0

    for video in fho["videos"]:
        video_uid = video["video_uid"]
        for interval in video["annotated_intervals"]:
            for action in interval["narrated_actions"]:
                total_actions += 1

                # Skip invalid or rejected actions
                if not action.get("is_valid_action", False):
                    continue
                if action.get("is_rejected", True):
                    continue

                # Must have critical_frames with a contact_frame
                critical = action.get("critical_frames")
                if not critical:
                    continue
                contact_frame = critical.get("contact_frame")
                if contact_frame is None:
                    continue

                kept_actions += 1
                video_actions[video_uid].append(
                    {
                        "start_frame": action["start_frame"],
                        "end_frame": action["end_frame"],
                        "contact_frame": contact_frame,
                        "narration_text": action.get("narration_text", ""),
                    }
                )

    print(
        f"  {kept_actions}/{total_actions} actions kept "
        f"({len(video_actions)} unique videos)"
    )
    return dict(video_actions)


def compute_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Return number of overlapping frames between [a_start, a_end] and [b_start, b_end]."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def find_best_action(
    vitra_frames: np.ndarray,
    actions: list[dict],
    min_overlap_ratio: float,
) -> dict | None:
    """
    Find the Ego4D action with maximum frame overlap with the VITRA episode frames.

    vitra_frames: sorted array of absolute Ego4D frame indices covered by the episode
    actions: list of action dicts for the same video_uid

    Returns the best action dict (with overlap_ratio added), or None if no action
    meets the minimum overlap ratio.
    """
    if len(vitra_frames) == 0 or not actions:
        return None

    ep_start = int(vitra_frames[0])
    ep_end = int(vitra_frames[-1])
    ep_len = ep_end - ep_start + 1

    best_action = None
    best_overlap = 0
    best_ratio = 0.0

    for action in actions:
        a_start = action["start_frame"]
        a_end = action["end_frame"]
        action_len = max(1, a_end - a_start + 1)

        overlap = compute_overlap(ep_start, ep_end, a_start, a_end)
        if overlap == 0:
            continue

        # Overlap ratio = overlap / min(episode_len, action_len)
        ratio = overlap / min(ep_len, action_len)

        if ratio > best_ratio or (ratio == best_ratio and overlap > best_overlap):
            best_overlap = overlap
            best_ratio = ratio
            best_action = action

    if best_action is None or best_ratio < min_overlap_ratio:
        return None

    return {**best_action, "overlap_ratio": best_ratio}


def contact_frame_to_local(
    contact_frame_ego4d: int, vitra_frames: np.ndarray,
    max_distance: int = 30,
) -> int | None:
    """
    Map an Ego4D absolute contact_frame to a local index within vitra_frames.

    Finds the index in vitra_frames whose value is closest to contact_frame_ego4d.
    Rejects (returns None) when the nearest VITRA frame is further than
    max_distance (default 30 ≈ 1 sec at 30 fps) from the target contact
    frame — this prevents silent clamping to frame 0 / T-1 when the Ego4D
    contact_frame is outside the VITRA episode's frame range.
    """
    if len(vitra_frames) == 0:
        return None
    diffs = np.abs(vitra_frames - contact_frame_ego4d)
    idx = int(np.argmin(diffs))
    if int(diffs[idx]) > max_distance:
        return None
    return idx


def collect_episode_files(vitra_root: str, subsets: list[str]) -> list[Path]:
    """Collect all .npy episode annotation files across the given subsets."""
    files = []
    for subset in subsets:
        ann_dir = Path(vitra_root) / subset / "episodic_annotations"
        if not ann_dir.exists():
            ann_dir = Path(vitra_root) / subset / subset / "episodic_annotations"
        if not ann_dir.exists():
            print(f"Warning: annotations directory not found: {ann_dir}", file=sys.stderr)
            continue
        npy_files = sorted(ann_dir.glob("*.npy"))
        print(f"  {subset}: {len(npy_files)} episodes")
        files.extend(npy_files)
    return files


def main():
    args = parse_args()

    # Load Ego4D action index
    video_actions = load_fho_main(args.fho_main)

    # Collect VITRA episode files
    print(f"\nCollecting VITRA episode files from subsets: {args.subsets}")
    episode_files = collect_episode_files(args.vitra_root, args.subsets)
    print(f"Total episodes found: {len(episode_files)}")

    # Build index
    index: dict[str, dict] = {}
    skipped_no_video = 0
    skipped_no_overlap = 0
    matched = 0

    for npy_path in tqdm(episode_files, desc="Building index"):
        stem = npy_path.stem

        try:
            episode = np.load(str(npy_path), allow_pickle=True).item()
        except Exception as e:
            print(f"Warning: failed to load {npy_path}: {e}", file=sys.stderr)
            continue

        video_uid = str(episode["video_name"])
        vitra_frames = np.asarray(episode["video_decode_frame"], dtype=np.int64)

        # Look up actions for this video
        actions = video_actions.get(video_uid)
        if not actions:
            skipped_no_video += 1
            continue

        # Find best matching action
        best = find_best_action(vitra_frames, actions, args.min_overlap_ratio)
        if best is None:
            skipped_no_overlap += 1
            continue

        # Map contact frame to local index
        local_idx = contact_frame_to_local(best["contact_frame"], vitra_frames)
        if local_idx is None:
            skipped_no_overlap += 1
            continue

        index[stem] = {
            "contact_frame_local": local_idx,
            "contact_frame_ego4d": int(best["contact_frame"]),
            "narration": best["narration_text"],
            "ego4d_start": int(best["start_frame"]),
            "ego4d_end": int(best["end_frame"]),
            "overlap_ratio": round(float(best["overlap_ratio"]), 4),
        }
        matched += 1

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    print("\nResults:")
    print(f"  Matched:              {matched}")
    print(f"  Skipped (no video):   {skipped_no_video}")
    print(f"  Skipped (no overlap): {skipped_no_overlap}")
    print(f"  Output written to:    {output_path}")


if __name__ == "__main__":
    main()
