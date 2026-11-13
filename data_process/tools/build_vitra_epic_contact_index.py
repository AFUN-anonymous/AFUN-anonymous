#!/usr/bin/env python3
"""Build the VITRA EPIC episode -> contact frame index (config: vitra_epic.yaml, epic_contact_index).

Uses the EPIC-KITCHENS-100 hand-object detections (Shan et al. pkls, read with
``epic_kitchens.hoa`` from epic-kitchens-100-hand-object-bboxes). For each VITRA
epic episode:
  1. Load video_name, video_decode_frame, anno_type (active hand).
  2. Locate the pkl at {hoa-root}/{Pxx}/hand-objects/{video_name}.pkl.
  3. Find the FIRST frame in [fs..fe] where the active hand has
     contact_state in {PORTABLE_OBJECT, STATIONARY_OBJECT}.
  4. Record contact_frame_local (offset within the episode) and metadata.

Episodes without a match are written as null; VITRAPreprocess skips them.

Usage:
    python tools/build_vitra_epic_contact_index.py \
        --vitra-root /path/to/VITRA-1M --hoa-root /path/to/hand-objects-root \
        --out /path/to/VITRA-1M/vitra_epic_contact_index.json
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

# CRITICAL: must be set BEFORE importing epic_kitchens.hoa (types_pb2 is
# incompatible with new google.protobuf descriptor API; pure-python parser
# is the zero-env-modification workaround).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from tqdm import tqdm

from epic_kitchens.hoa import load_detections
from epic_kitchens.hoa.types import HandState, HandSide

CONTACT_STATES = {HandState.PORTABLE_OBJECT, HandState.STATIONARY_OBJECT}

VITRA_NAME_RE = re.compile(r"epic_kitchens_(P\d+_\d+)_ep_\d+\.npy")


def side_match(det_side, vitra_side: str) -> bool:
    if vitra_side == "right":
        return det_side == HandSide.RIGHT
    if vitra_side == "left":
        return det_side == HandSide.LEFT
    return False


def find_contact(detections, fs: int, fe: int, vitra_side: str):
    """Return (global_frame, state_name) of first contact, or (None, None).

    `detections` is the parsed list (one entry per source-video frame).
    `fs..fe` are absolute source-video frame indices (inclusive).
    """
    n = len(detections)
    for f in range(fs, min(fe + 1, n)):
        det = detections[f]
        for h in det.hands:
            if side_match(h.side, vitra_side) and h.state in CONTACT_STATES:
                return f, h.state.name
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitra-root", required=True,
                    help="VITRA-1M root (contains epic/epic/episodic_annotations/)")
    ap.add_argument("--hoa-root", required=True,
                    help="Root containing {Pxx}/hand-objects/*.pkl")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    vitra_files = sorted(Path(args.vitra_root).glob("epic/epic/episodic_annotations/*.npy"))

    print(f"[idx] scanning {len(vitra_files)} VITRA epic episodes", flush=True)

    # Sort by video_name so we only keep one pkl in memory at a time.
    def sort_key(p):
        m = VITRA_NAME_RE.match(p.name)
        return (m.group(1) if m else "", p.name)
    vitra_files = sorted(vitra_files, key=sort_key)

    out: dict = {}
    matched = 0
    no_pkl = 0
    no_contact = 0
    unreadable = 0

    cur_video = None
    cur_dets = None

    state_counter = Counter()

    for fp in tqdm(vitra_files, desc="episodes"):
        stem = fp.stem  # e.g. epic_kitchens_P01_01_ep_000000
        try:
            d = np.load(fp, allow_pickle=True).item()
        except Exception as e:
            print(f"[warn] failed to load {fp.name}: {e}", flush=True)
            out[stem] = None
            unreadable += 1
            continue

        vid = d["video_name"]
        frames = d["video_decode_frame"]
        fs = int(frames[0])
        fe = int(frames[-1])
        side = d.get("anno_type", "right")

        # Load pkl if we've moved to a new video
        if vid != cur_video:
            cur_video = vid
            cur_dets = None
            pid = vid.split("_")[0]
            pkl_path = Path(args.hoa_root) / pid / "hand-objects" / f"{vid}.pkl"
            if not pkl_path.exists():
                # Try fallback: some layouts nest differently
                pkl_path = Path(args.hoa_root) / pid / f"{vid}.pkl"
            if not pkl_path.exists():
                print(f"[miss] no pkl for {vid}", flush=True)
            else:
                try:
                    cur_dets = list(load_detections(str(pkl_path)))
                except Exception as e:
                    print(f"[warn] failed to parse {pkl_path}: {e}", flush=True)
                    cur_dets = None

        if cur_dets is None:
            out[stem] = None
            no_pkl += 1
            continue

        cf, state = find_contact(cur_dets, fs, fe, side)
        if cf is None:
            out[stem] = None
            no_contact += 1
            continue

        out[stem] = {
            "video_name": vid,
            "hand_side": side,
            "contact_frame_global": cf,
            "contact_frame_local": cf - fs,
            "contact_state": state,
            "episode_start_global": fs,
            "episode_end_global": fe,
        }
        state_counter[state] += 1
        matched += 1

    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(
        f"\n[idx] DONE  matched={matched}  no_pkl={no_pkl}  "
        f"no_contact={no_contact}  unreadable={unreadable}  "
        f"total={len(vitra_files)}",
        flush=True,
    )
    print(f"[idx] contact_state distribution: {dict(state_counter)}", flush=True)
    print(f"[idx] wrote: {args.out}", flush=True)


if __name__ == "__main__":
    main()
