#!/usr/bin/env python3
"""Build HTML pages for the manual Keep/Discard review of finished pipeline samples.

    python tools/build_review.py {output_root} {out_dir} [--datasets droid calvin]

Writes ``<out_dir>/<dataset>_part001.html``. Each sample shows its task, the observation with the
mask, the tracked motion (yellow), the fitted Bezier curve (green) and a 3D view of the scene.
Serve the folder with ``python -m http.server`` so all pages share their decisions in the browser.
The "Keep list" button downloads the ids for ``tools/export_dataset.py --keep-list``.

Samples need a fitted curve (``fsparams`` and ``spline_params``), so SceneFun3D, whose motion comes
from annotations, is not reviewed here.
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from export_dataset import finished_cams, language_of, mask_of, read_json

TEMPLATE = Path(__file__).with_name("review_template.html")
NO_FIT_FILTER = {"calvin"}  # reviewed whatever their curve-fit error
MAX_RANGE_M = 5.0


def encode(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def scaled_intrinsics(intrinsics, width, height):
    """(fx, fy, cx, cy) rescaled from the calibrated resolution to width x height."""
    calib_width, calib_height = (intrinsics.get("resolution") or (width, height))[:2]
    sx, sy = width / calib_width, height / calib_height
    cx, cy = intrinsics.get("cx", intrinsics.get("ppx")), intrinsics.get("cy", intrinsics.get("ppy"))
    return float(intrinsics["fx"]) * sx, float(intrinsics["fy"]) * sy, float(cx) * sx, float(cy) * sy


def bezier_curve(r0, ctrl, num_points=50):
    """Sample the Bezier with P0 = r0 and control points given relative to r0."""
    r0 = np.asarray(r0, dtype=np.float64)
    points = np.vstack([r0, r0 + np.asarray(ctrl, dtype=np.float64).reshape(-1, 3)])
    degree = len(points) - 1
    t = np.linspace(0, 1, num_points)[:, None]
    return sum(math.comb(degree, i) * t**i * (1 - t) ** (degree - i) * p for i, p in enumerate(points))


def draw_observation(rgb, mask, track, curve, camera):
    """Observation with the red mask, its yellow box, the 2D track and the projected Bezier."""
    blended = rgb.astype(np.float32)
    blended[mask] = blended[mask] * 0.55 + np.array([255.0, 0.0, 0.0]) * 0.45
    image = Image.fromarray(blended.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    if mask.any():
        ys, xs = np.nonzero(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        draw.rectangle(box, outline=(255, 255, 0), width=3)
    if len(track) > 1:
        draw.line(track, fill=(255, 220, 0), width=5)
    for x, y in track:
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 220, 0))
    fx, fy, cx, cy = camera
    pixels = [(fx * x / z + cx, fy * y / z + cy) for x, y, z in curve if z > 0]
    if len(pixels) > 1:
        draw.line(pixels, fill=(0, 255, 68), width=7)
    image.thumbnail((480, 480))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_centroid(mask, depth, camera):
    """3D point at the mask's pixel centroid and median depth, or None without valid depth."""
    depths = depth[mask]
    depths = depths[depths > 0]
    if depths.size == 0:
        return None
    fx, fy, cx, cy = camera
    ys, xs = np.nonzero(mask)
    z = float(np.median(depths)) / 1000
    return np.array([(xs.mean() - cx) * z / fx, (ys.mean() - cy) * z / fy, z])


def point_cloud(rgb, mask, depth, camera, num_points):
    """Randomly sampled scene points within MAX_RANGE_M: (xyz, rgb, in_mask)."""
    fx, fy, cx, cy = camera
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs].astype(np.float64) / 1000
    xyz = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    keep = np.flatnonzero(np.linalg.norm(xyz, axis=1) <= MAX_RANGE_M)
    if len(keep) > num_points:
        keep = np.random.default_rng(42).choice(keep, num_points, replace=False)
    return xyz[keep], rgb[ys[keep], xs[keep]], mask[ys[keep], xs[keep]]


def load_sample(cam, sample_id, max_rel_err, num_points):
    """Page data of one cam, or None if the cam cannot be reviewed."""
    traj = read_json(cam / "trajectory.json") or {}
    fit, spline = traj.get("fsparams") or {}, traj.get("spline_params") or {}
    rel_err = fit.get("shape_fit_relative_error")
    mask_path, language = mask_of(cam), language_of(cam)
    if "r0" not in fit or "ctrl" not in spline or rel_err is None or rel_err >= max_rel_err:
        return None
    has_files = (cam / "obs_frame.png").exists() and (cam / "obs_frame_depth.npy").exists()
    if not (mask_path and language and has_files):
        return None

    image = Image.open(cam / "obs_frame.png").convert("RGB")
    mask = Image.open(mask_path).convert("L")
    depth = np.load(cam / "obs_frame_depth.npy").astype(np.float32)
    depth_size = depth.shape[1], depth.shape[0]
    intrinsics = traj["camera_info"]["intrinsics"]

    curve = bezier_curve(fit["r0"], spline["ctrl"])
    raw = np.array([p["position_3d"] for p in traj.get("trajectory_3d") or []], dtype=np.float64).reshape(-1, 3)
    track_2d = (read_json(cam / "sam3_mask_traj.json") or {}).get("trajectory_2d") or []
    track = [(int(p["position_pixel"][0]), int(p["position_pixel"][1])) for p in track_2d]
    observation = draw_observation(np.array(image), np.array(mask.resize(image.size, Image.NEAREST)) > 127,
                                   track, curve, scaled_intrinsics(intrinsics, *image.size))

    # The 3D view starts both curves at the mask, where the model anchors its prediction.
    camera = scaled_intrinsics(intrinsics, *depth_size)
    depth_mask = np.array(mask.resize(depth_size, Image.NEAREST)) > 127
    anchor = mask_centroid(depth_mask, depth, camera)
    if anchor is not None:
        curve = curve + anchor - curve[0]
        raw = raw + anchor - raw[0] if len(raw) else raw
    xyz, colors, in_mask = point_cloud(np.array(image.resize(depth_size, Image.BILINEAR)), depth_mask, depth,
                                       camera, num_points)
    return {"id": sample_id, "dataset": sample_id.split("/")[0], "language": language, "rel_err": float(rel_err),
            "quality": fit.get("shape_fit_quality", "unknown"), "image": observation,
            "xyz": encode(xyz.ravel(), np.float32), "rgb": encode(colors.ravel(), np.uint8),
            "in_mask": encode(in_mask, np.uint8), "raw": encode(raw.ravel(), np.float32),
            "curve": encode(curve.ravel(), np.float32)}


def write_page(path, samples):
    data = json.dumps(samples, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__TITLE__", path.stem).replace("__SAMPLES__", data)
    path.write_text(html, encoding="utf-8")
    print(f"{path}: {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_root", type=Path, help="pipeline output root (holds the dataset folders)")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--datasets", nargs="+", help="dataset folders to review (default: all)")
    parser.add_argument("--max-rel-err", type=float, default=0.08, help="largest curve-fit error to review")
    parser.add_argument("--pc-points", type=int, default=10000, help="scene points in each 3D view")
    parser.add_argument("--page-size", type=int, default=1000, help="samples per HTML page")
    args = parser.parse_args()

    root = args.output_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets or sorted(d.name for d in root.iterdir() if d.is_dir()):
        if not (root / dataset).is_dir():
            print(f"skipping {dataset}: no {root / dataset}")
            continue
        max_rel_err = math.inf if dataset in NO_FIT_FILTER else args.max_rel_err
        page, part = [], 0
        for key in tqdm(finished_cams(root, dataset), desc=dataset):
            sample = load_sample(root / dataset / key, f"{dataset}/{key}", max_rel_err, args.pc_points)
            if sample:
                page.append(sample)
            if len(page) == args.page_size:
                part += 1
                write_page(args.out_dir / f"{dataset}_part{part:03d}.html", page)
                page = []
        if page:
            write_page(args.out_dir / f"{dataset}_part{part + 1:03d}.html", page)


if __name__ == "__main__":
    main()
