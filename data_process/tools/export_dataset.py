#!/usr/bin/env python3
"""Export finished pipeline samples into the released AFUN layout.

    python tools/export_dataset.py {output_root} {out_dir} [--datasets droid calvin] [--keep-list ids.txt]

Writes ``<out>/<dataset>/<episode>/<interval>/<cam>/`` with obs_frame.png,
obs_frame_depth.npy (float32, mm), sam_mask.png and trajectory.json, plus
``<out>/manifest.json``. The source tree is only read.

A cam is exported when, as in the AFUN release build:
  * ``{output_root}/sam3_mask_traj_manifest.json`` marks it done with num_points > 0
    (scenefun3d: ``{output_root}/scenefun3d_sam_match_manifest.json`` marks it valid);
  * obs_frame.png, obs_frame_depth.npy and trajectory.json exist;
  * metadata.json has a non-empty ``interval_language`` (cam level first, then interval);
  * it has a mask for the observation frame: when ``sam3_text_prompt_tracking_masks/``
    exists, its ``frame_00000.png`` (tracking indices start at obs_frame; if the first
    tracked frame is later the mask does not match obs_frame.png and the cam is skipped),
    otherwise the cam's ``sam_mask.png``;
  * after removing DROP_KEYS, trajectory.json holds exactly TRAJ_KEYS (so spline_params
    is present), or, for scenefun3d, camera_info plus a ``trajectories`` list whose entries
    all hold SF3D_TRAJ_KEYS. fsparams must go: the AFUN loader prefers it over spline_params.
Ego4D-derived sources (vitra_ego4d*) were released as annotations only, without frames.
"""
import argparse
import datetime
import json
import os
import shutil
from collections import Counter
from pathlib import Path

MANIFEST_NAME = "sam3_mask_traj_manifest.json"
SF3D_MANIFEST_NAME = "scenefun3d_sam_match_manifest.json"
DROP_KEYS = {"fsparams", "trajectory_2d", "raw_trajectory_3d", "raw_trajectory_2d",
             "motion_3d", "contact_flange_point"}
TRAJ_KEYS = {"camera_info", "motion_2d", "spline_params", "trajectory_3d"}
SF3D_TRAJ_KEYS = {"motion_2d", "spline_params", "trajectory_3d"}
FILES = ["obs_frame.png", "obs_frame_depth.npy", "sam_mask.png", "trajectory.json"]


def strip(obj):
    """Drop DROP_KEYS at every nesting level."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def language_of(cam):
    for path in (cam / "metadata.json", cam.parent / "metadata.json"):
        meta = read_json(path)
        if isinstance(meta, dict) and meta.get("interval_language"):
            return meta["interval_language"]
    return None


def mask_of(cam):
    tracking = cam / "sam3_text_prompt_tracking_masks"
    mask = tracking / "frame_00000.png" if tracking.is_dir() else cam / "sam_mask.png"
    return mask if mask.exists() else None


def valid_trajectory(traj):
    if not isinstance(traj, dict):
        return False
    if set(traj) == {"camera_info", "trajectories"}:  # scenefun3d: one entry per annotation
        trajs = traj["trajectories"]
        return bool(trajs) and all(isinstance(t, dict) and SF3D_TRAJ_KEYS <= set(t) for t in trajs)
    return set(traj) == TRAJ_KEYS


def finished_cams(root, dataset):
    """Keys <episode>/<interval>/<cam> of the dataset's cams that finished the pipeline."""
    if dataset == "scenefun3d":
        manifest = read_json(root / SF3D_MANIFEST_NAME) or {}
        keys = []
        for key, v in manifest.items():  # key: <episode>__<interval idx>__<cam>
            parts = key.split("__")
            if len(parts) == 3 and isinstance(v, dict) and v.get("status") == "valid":
                keys.append(f"{parts[0]}/interval_{parts[1]}/{parts[2]}")
        return sorted(keys)
    manifest = read_json(root / MANIFEST_NAME) or {}
    episodes = set(os.listdir(root / dataset))
    return sorted(k for k, v in manifest.items() if len(k.split("/")) == 3 and k.split("/")[0] in episodes
                  and isinstance(v, dict) and v.get("status") == "done" and (v.get("num_points") or 0) > 0)


def prepare(cam):
    """Return ("ok", (mask, language, trajectory)) or (skip reason, None)."""
    if not all((cam / name).exists() for name in ("obs_frame.png", "obs_frame_depth.npy", "trajectory.json")):
        return "missing_files", None
    language = language_of(cam)
    if not language:
        return "no_language", None
    mask = mask_of(cam)
    if mask is None:
        return "no_obs_frame_mask", None
    traj = strip(read_json(cam / "trajectory.json"))
    if not valid_trajectory(traj):
        return "bad_trajectory_keys", None
    return "ok", (mask, language, traj)


def place(src, dst, hardlink):
    if dst.exists():
        dst.unlink()
    if hardlink:
        os.link(src, dst)
    else:
        shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_root", type=Path, help="pipeline output root (holds the dataset folders)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--datasets", nargs="+", help="dataset folders to export (default: all)")
    parser.add_argument("--keep-list", type=Path,
                        help="text file of sample ids <dataset>/<episode>/<interval>/<cam>; export only these")
    parser.add_argument("--hardlink", action="store_true", help="hardlink instead of copying (same filesystem)")
    args = parser.parse_args()

    root, out = args.output_root.resolve(), args.out_dir.resolve()
    datasets = args.datasets or sorted(d.name for d in root.iterdir() if d.is_dir())
    keep = set(args.keep_list.read_text().split()) if args.keep_list else None

    rows, reasons = [], Counter()
    for dataset in datasets:
        if not (root / dataset).is_dir():
            print(f"skipping {dataset}: no {root / dataset}")
            continue
        for key in finished_cams(root, dataset):
            episode, interval, cam_name = key.split("/")
            sample_id = f"{dataset}/{key}"
            if keep is not None and sample_id not in keep:
                continue
            cam = root / dataset / key
            status, payload = prepare(cam) if cam.is_dir() else ("missing_cam_dir", None)
            reasons[status] += 1
            if status != "ok":
                continue
            mask, language, traj = payload
            dst = out / sample_id
            dst.mkdir(parents=True, exist_ok=True)
            place(cam / "obs_frame.png", dst / "obs_frame.png", args.hardlink)
            place(cam / "obs_frame_depth.npy", dst / "obs_frame_depth.npy", args.hardlink)
            place(mask, dst / "sam_mask.png", args.hardlink)
            (dst / "trajectory.json").write_text(json.dumps(traj, separators=(",", ":")))
            rows.append({"sample_id": sample_id, "path": sample_id, "dataset": dataset,
                         "episode_id": episode, "interval": interval, "cam": cam_name, "language": language})

    rows.sort(key=lambda r: (r["dataset"], r["episode_id"], r["interval"], r["cam"]))
    per_dataset = dict(Counter(r["dataset"] for r in rows).most_common())
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.json", "w") as f:
        json.dump({"name": "AFUN", "generated": datetime.date.today().isoformat(), "files_per_sample": FILES,
                   "total": len(rows), "per_dataset": per_dataset, "samples": rows}, f, ensure_ascii=False, indent=1)
    if keep is not None and len(rows) < len(keep):
        print(f"keep-list: {len(keep) - len(rows)} of {len(keep)} ids were not exported")
    print(f"checked: {dict(reasons)}")
    print(f"exported {len(rows)} samples to {out}: {per_dataset}")


if __name__ == "__main__":
    main()
