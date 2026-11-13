#!/usr/bin/env python3
"""Fit Bezier motion targets for every finished cam under one dataset.

Usage (from the repo root):
    python curve_fit/run_curve_fit_dataset.py {output_root}/{dataset} [--skip-existing]

Cams are taken from ``{output_root}/sam3_mask_traj_manifest.json`` (written by
the sam3_mask_traj / sam3_mask_traj_depth stages): every entry with status
"done" whose ``{dataset}/<episode>/<interval>/<cam>/sam3_mask_traj.json`` exists.
Each fit merges fsparams, spline_params and the fitted trajectory into that
cam's ``trajectory.json`` (see fit_sam3_constant_curvature.process_cam_dir).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS/OpenMP thread per worker process (set before numpy is imported).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from tqdm import tqdm  # noqa: E402

from fit_sam3_constant_curvature import process_cam_dir  # noqa: E402

MANIFEST_NAME = "sam3_mask_traj_manifest.json"  # mainprocess.utils.manifest_path(root, STAGE_SAM3_MASK_TRAJ)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="{output_root}/{dataset}")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip cams whose trajectory.json already has spline_params")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N cams (sorted), for testing")
    parser.add_argument("--num-workers", type=int, default=12)
    return parser.parse_args()


def discover_cam_dirs(dataset_root, limit=None):
    """Cams marked done in the sam3_mask_traj manifest that live under dataset_root."""
    manifest_path = dataset_root.parent / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found: run the sam3_mask_traj stage first.")
    manifest = json.loads(manifest_path.read_text())
    cam_dirs = []
    for key in sorted(k for k, v in manifest.items() if isinstance(v, dict) and v.get("status") == "done"):
        if len(key.split("/")) != 3:
            continue
        cam_dir = dataset_root / key
        if (cam_dir / "sam3_mask_traj.json").exists():
            cam_dirs.append(cam_dir)
            if limit and len(cam_dirs) >= limit:
                break
    return cam_dirs


def _fit_one(cam_dir, skip_existing):
    try:
        traj_path = cam_dir / "trajectory.json"
        if skip_existing and traj_path.exists() and "spline_params" in json.loads(traj_path.read_text()):
            return cam_dir, "skipped", ""
        process_cam_dir(cam_dir)
        return cam_dir, "ok", ""
    except Exception as exc:
        return cam_dir, "error", f"{type(exc).__name__}: {exc}"


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    cam_dirs = discover_cam_dirs(dataset_root, limit=args.limit)
    print(f"Dataset root: {dataset_root}\nCam folders : {len(cam_dirs)}", flush=True)
    if not cam_dirs:
        return

    counts = {"ok": 0, "skipped": 0, "error": 0}
    num_workers = args.num_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = pool.map(_fit_one, cam_dirs, [args.skip_existing] * len(cam_dirs), chunksize=16)
        for cam_dir, status, error in tqdm(results, total=len(cam_dirs), desc="Curve fit", unit="cam"):
            counts[status] += 1
            if status == "error":
                tqdm.write(f"  ERROR {cam_dir.relative_to(dataset_root)}: {error}")

    print(f"Done: {counts['ok']} fitted, {counts['skipped']} skipped, {counts['error']} failed "
          f"(of {len(cam_dirs)})", flush=True)


if __name__ == "__main__":
    main()
