"""Build the episode list for vitra_ego4d_noannot (config: episode_txt_list).

Keeps the VITRA Ego4D episodes that are NOT in the Ego4D contact index (those
go through vitra_ego4d) and whose source video exists in the Ego4D video folder.
Writes one .npy path per line, sorted.

Usage:
    python tools/build_vitra_ego4d_noannot_list.py \\
        --vitra-root /path/to/VITRA-1M \\
        --contact-index /path/to/VITRA-1M/ego4d_contact_index_full.json \\
        --video-root /path/to/Ego4D/v2/full_scale \\
        --out /path/to/VITRA-1M/ego4d_noannot_episodes.txt
"""

import argparse
import glob
import json
import os
from pathlib import Path

SUBSETS = ("ego4d_cooking_and_cleaning", "ego4d_other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vitra-root", required=True)
    ap.add_argument("--contact-index", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    idx = set(json.load(open(args.contact_index)))
    avail = set(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.video_root, "*.mp4")))

    paths = []
    for sub in SUBSETS:
        d = Path(args.vitra_root) / sub / "episodic_annotations"
        if not d.is_dir():  # nested layout, as handled by VITRAPreprocess
            d = Path(args.vitra_root) / sub / sub / "episodic_annotations"
        for f in d.iterdir():
            if not f.name.endswith(".npy"):
                continue
            stem = f.name[:-4]
            if stem in idx:
                continue
            vuid = stem.replace("Ego4D_", "").rsplit("_ep_", 1)[0]
            if vuid in avail:
                paths.append(str(f))
    paths.sort()
    Path(args.out).write_text("\n".join(paths) + "\n")
    print(f"wrote {len(paths)} episodes to {args.out}")


if __name__ == "__main__":
    main()
