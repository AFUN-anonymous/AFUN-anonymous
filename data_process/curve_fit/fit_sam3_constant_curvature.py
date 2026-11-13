#!/usr/bin/env python3
"""Robust planar constant-curvature fit of a back-projected mask-centroid trajectory.

Paper Appendix C.2, "Curve fit". Steps, in ``fit_constant_curvature``:
  1. down-weight abnormally large temporal steps (median + 4 * MAD rule);
  2. smooth local windows with a weighted geometric median;
  3. resample into support points that are uniform in cumulative arc length,
     weighted by the inverse square of their local spread;
  4. fit x(s) = p0 + s * sinc(k s) e1 + s * cosc(k s) e2 by alternating monotone
     arc-length assignment (isotonic regression) and nonlinear least squares
     with a Cauchy loss;
  5. snap to a straight line when |k| * arc_length < LINE_THRESHOLD_RAD, or when
     the line's weighted RMS is within PREFER_LINE_RATIO of the arc's.

``process_cam_dir`` fits ``sam3_mask_traj.json`` and merges the result
(fsparams, spline_params, fitted trajectory) into the cam's ``trajectory.json``.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from convert_cc_to_fsframe import build_fitted_trajectory, save_to_trajectory_json

EPS = 1e-8
ITERATIONS = 12            # max alternating (s-assignment, least-squares) rounds
GRID_SIZE = 512            # samples used to find the closest curve point
CAUCHY_SCALE = 0.03        # Cauchy f_scale floor, in units of the chord length
LINE_THRESHOLD_RAD = 0.15  # snap to a line when |kappa| * arc_length is below this
PREFER_LINE_RATIO = 1.2    # also snap when line RMS <= 1.2 * arc RMS
SUPPORT_BANDWIDTH_SCALE = 1.5
SUPPORT_MIN_NEIGHBORS = 3
QUALITY_THRESHOLDS = (("excellent", 0.025), ("good", 0.065), ("fair", 0.125))


def normalize(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return v * 0.0
    return v / n


def orthogonal_vector(v):
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(v, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return normalize(np.cross(v, basis))


def orthonormal_frame(t_raw, n_raw):
    e1 = normalize(np.asarray(t_raw, dtype=np.float64))
    if np.linalg.norm(e1) < EPS:
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n = np.asarray(n_raw, dtype=np.float64) - np.dot(n_raw, e1) * e1
    if np.linalg.norm(n) < EPS:
        n = orthogonal_vector(e1)
    else:
        n = normalize(n)
    e2 = normalize(np.cross(n, e1))
    return e1, e2, n


def sinc(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.ones_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - xs**2 / 6.0 + xs**4 / 120.0
    out[~small] = np.sin(x[~small]) / x[~small]
    return out


def cosc(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = xs / 2.0 - xs**3 / 24.0 + xs**5 / 720.0
    out[~small] = (1.0 - np.cos(x[~small])) / x[~small]
    return out


def curve_points(p0, e1, e2, kappa, s):
    s = np.asarray(s, dtype=np.float64)
    ks = kappa * s
    term1 = (s * sinc(ks))[:, None] * e1[None, :]
    term2 = (s * cosc(ks))[:, None] * e2[None, :]
    return p0[None, :] + term1 + term2


def weighted_pca_line(points, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, EPS)
    center = np.average(points, axis=0, weights=weights)
    centered = points - center
    cov = centered.T @ (centered * weights[:, None])
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]
    return center, normalize(direction)


def estimate_plane_normal(points, weights, direction):
    weights = np.asarray(weights, dtype=np.float64)
    center = np.average(points, axis=0, weights=np.maximum(weights, EPS))
    centered = points - center
    cov = centered.T @ (centered * weights[:, None])
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, np.argmin(vals)]
    normal = normal - np.dot(normal, direction) * direction
    if np.linalg.norm(normal) < EPS:
        normal = orthogonal_vector(direction)
    return normalize(normal)


def weighted_pava(values, weights):
    """Weighted isotonic (non-decreasing) regression: keeps arc length monotone."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    blocks = []
    for value, weight in zip(values, weights):
        blocks.append([float(value), float(weight), 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v1, w1, c1 = blocks[-2]
            v2, w2, c2 = blocks[-1]
            merged_w = w1 + w2
            merged_v = (v1 * w1 + v2 * w2) / max(merged_w, EPS)
            blocks[-2:] = [[merged_v, merged_w, c1 + c2]]
    out = np.empty_like(values)
    idx = 0
    for value, _, count in blocks:
        out[idx:idx + count] = value
        idx += count
    return out


def mark_suspicious_steps(points):
    """Flag points reached by an abnormally large step (tracking jumps)."""
    if len(points) < 3:
        return np.zeros(len(points), dtype=bool)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)))
    sigma = 1.4826 * mad
    if sigma < EPS:
        suspicious_steps = deltas > (med * 4.0 if med > EPS else 0.0)
    else:
        suspicious_steps = deltas > med + 4.0 * sigma
    suspicious = np.zeros(len(points), dtype=bool)
    suspicious[1:] = suspicious_steps
    return suspicious


def fit_circle_2d(points_2d, weights):
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points_2d))))
    b = x**2 + y**2
    w = np.sqrt(np.maximum(weights, EPS))[:, None]
    params, _, _, _ = np.linalg.lstsq(a * w, b * w[:, 0], rcond=None)
    cx, cy, c = params
    radius_sq = c + cx**2 + cy**2
    if radius_sq <= EPS:
        return None
    radius = math.sqrt(radius_sq)
    residuals = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)
    return np.array([cx, cy], dtype=np.float64), float(radius), residuals


def initial_curvature(points, weights, line_center, line_dir, plane_normal):
    e1 = line_dir
    e2 = normalize(np.cross(plane_normal, e1))
    rel = points - line_center
    pts2d = np.column_stack((rel @ e1, rel @ e2))
    fitted = fit_circle_2d(pts2d, weights)
    if fitted is None:
        return 0.0
    center_2d, radius, residuals = fitted
    extent = max(np.linalg.norm(pts2d[-1] - pts2d[0]), EPS)
    if radius > 20.0 * extent:
        return 0.0
    poly = pts2d - center_2d[None, :]
    signed_turn = float(np.sum(poly[:-1, 0] * poly[1:, 1] - poly[:-1, 1] * poly[1:, 0]))
    sign = 1.0 if signed_turn >= 0 else -1.0
    if float(np.median(residuals)) > 0.2 * extent:
        return 0.0
    return sign / max(radius, EPS)


def robust_sigma(residual_norms, fallback):
    residual_norms = np.asarray(residual_norms, dtype=np.float64)
    if len(residual_norms) == 0:
        return fallback
    med = float(np.median(residual_norms))
    mad = float(np.median(np.abs(residual_norms - med)))
    sigma = 1.4826 * mad
    return max(sigma, fallback)


def weighted_rms(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = max(float(np.sum(weights)), EPS)
    return math.sqrt(float(np.sum(weights * values**2)) / denom)


def geometric_median(points, weights=None, max_iter=64):
    """Weighted geometric median (Weiszfeld iterations)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return points[0].copy()
    if weights is None:
        weights = np.ones(len(points), dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), EPS)
    guess = np.average(points, axis=0, weights=weights)
    for _ in range(max_iter):
        dists = np.linalg.norm(points - guess[None, :], axis=1)
        if float(np.min(dists)) < 1e-6:
            return points[int(np.argmin(dists))].copy()
        step_weights = weights / np.maximum(dists, 1e-6)
        updated = np.average(points, axis=0, weights=step_weights)
        if np.linalg.norm(updated - guess) < 1e-6:
            return updated
        guess = updated
    return guess


def robust_smooth_points(points, window_radius, weights):
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if window_radius <= 0 or len(points) < 3:
        return points.copy()
    smoothed = np.empty_like(points)
    for i in range(len(points)):
        lo = max(0, i - window_radius)
        hi = min(len(points), i + window_radius + 1)
        smoothed[i] = geometric_median(points[lo:hi], weights[lo:hi])
    return smoothed


def compute_cumulative_arclength(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def choose_support_count(num_points):
    """About one support point per 8 raw points, between 8 and 40."""
    if num_points <= 1:
        return 1
    return max(1, min(num_points, min(40, max(8, int(round(num_points / 8.0))))))


def build_support_points(points, smoothed_points, arclength, raw_weights):
    """Resample into ~uniform-arc-length support points.

    Returns (support_points, support_weights, arc-length targets). Each support
    point is the geometric median of its arc-length neighbourhood; its weight is
    1 / spread**2 (clipped), and outlying spreads are down-weighted 10x.
    """
    num_points = len(points)
    count = choose_support_count(num_points)
    total_length = float(arclength[-1]) if len(arclength) else 0.0
    targets = np.linspace(0.0, total_length, count) if count > 1 else np.array([0.0])
    spacing = total_length / max(count - 1, 1)
    base_bandwidth = total_length / max(num_points - 1, 1)
    bandwidth = max(SUPPORT_BANDWIDTH_SCALE * spacing, base_bandwidth, 1e-3)
    need = max(2, min(SUPPORT_MIN_NEIGHBORS, num_points))

    support_points = []
    spreads = []
    for target in targets:
        idx = np.flatnonzero(np.abs(arclength - target) <= bandwidth)
        if len(idx) < need:
            center = int(np.argmin(np.abs(arclength - target)))
            lo = max(0, center - need // 2)
            hi = min(num_points, lo + need)
            lo = max(0, hi - need)
            idx = np.arange(lo, hi)
        rep = geometric_median(smoothed_points[idx], raw_weights[idx])
        spreads.append(float(np.median(np.linalg.norm(points[idx] - rep[None, :], axis=1))))
        support_points.append(rep)

    support_points = np.asarray(support_points, dtype=np.float64)
    spreads = np.asarray(spreads, dtype=np.float64)
    positive = spreads[spreads > EPS]
    spread_floor = max(
        float(np.median(positive)) * 0.25 if len(positive) else 0.0,
        spacing * 0.25,
        1e-3,
    )
    support_weights = 1.0 / np.maximum(spreads, spread_floor) ** 2
    support_weights /= max(float(np.median(support_weights)), 1.0)
    support_weights = np.clip(support_weights, 0.1, 10.0)

    spread_med = float(np.median(spreads)) if len(spreads) else 0.0
    spread_mad = float(np.median(np.abs(spreads - spread_med))) if len(spreads) else 0.0
    spread_sigma = 1.4826 * spread_mad
    if spread_sigma < EPS:
        bad = spreads > max(3.0 * spread_med, spread_floor * 4.0)
    else:
        bad = spreads > spread_med + 3.0 * spread_sigma
    support_weights[bad] *= 0.1
    return support_points, support_weights, targets


def nearest_s(points, p0, e1, e2, kappa, current_s, weights):
    s_min = float(np.min(current_s))
    s_max = float(np.max(current_s))
    span = max(s_max - s_min, 1.0)
    grid = np.linspace(s_min - 0.2 * span, s_max + 0.2 * span, max(GRID_SIZE, 64))
    samples = curve_points(p0, e1, e2, kappa, grid)
    dists = np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2)
    best = grid[np.argmin(dists, axis=1)]
    return weighted_pava(best, weights)


def parameter_pack(p0, t_raw, n_raw, kappa):
    return np.concatenate([p0, t_raw, n_raw, [kappa]]).astype(np.float64)


def unpack_params(params):
    p0 = params[:3]
    e1, e2, n = orthonormal_frame(params[3:6], params[6:9])
    kappa = float(params[9])
    return p0, e1, e2, n, kappa


def fit_geometry(points, s_values, weights, params0, sigma):
    sqrt_w = np.sqrt(np.maximum(weights, EPS))

    def residuals(params):
        p0, e1, e2, _, kappa = unpack_params(params)
        pred = curve_points(p0, e1, e2, kappa, s_values)
        return ((pred - points) * sqrt_w[:, None]).ravel()

    return least_squares(
        residuals,
        params0,
        method="trf",
        loss="cauchy",
        f_scale=max(sigma, EPS),
        max_nfev=200,
    )


def evaluate_curve(points, weights, params, initial_s=None):
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    p0, e1, e2, _, kappa = unpack_params(params)
    if initial_s is None or len(initial_s) != len(points):
        initial_s = (points - p0[None, :]) @ e1
    s_values = weighted_pava(np.asarray(initial_s, dtype=np.float64), weights)
    s_values = nearest_s(points, p0, e1, e2, kappa, s_values, weights)
    pred = curve_points(p0, e1, e2, kappa, s_values)
    residuals = np.linalg.norm(pred - points, axis=1)
    return s_values, pred, residuals


def fit_constant_curvature_core(points, weights):
    points = np.asarray(points, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), EPS)
    if len(points) == 1:
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e2 = orthogonal_vector(e1)
        n = normalize(np.cross(e1, e2))
        return {
            "params": parameter_pack(points[0], e1, n, 0.0),
            "e1": e1,
            "e2": e2,
            "kappa": 0.0,
            "s_values": np.zeros(1, dtype=np.float64),
            "model_type": "line",
            "arc_length": 0.0,
        }

    line_center, line_dir = weighted_pca_line(points, weights)
    line_s = (points - line_center) @ line_dir
    if line_s[-1] < line_s[0]:
        line_dir = -line_dir
        line_s = -line_s
    line_s = weighted_pava(line_s, weights)

    plane_normal = estimate_plane_normal(points, weights, line_dir)
    kappa0 = initial_curvature(points, weights, line_center, line_dir, plane_normal)
    params = parameter_pack(line_center, line_dir, plane_normal, kappa0)
    sigma = CAUCHY_SCALE
    s_values = line_s

    last_cost = None
    for _ in range(ITERATIONS):
        p0, e1, e2, _, kappa = unpack_params(params)
        s_values = nearest_s(points, p0, e1, e2, kappa, s_values, weights)
        result = fit_geometry(points, s_values, weights, params, sigma)
        params = result.x
        s_values, pred, residuals = evaluate_curve(points, weights, params, s_values)
        sigma = robust_sigma(residuals, CAUCHY_SCALE)
        if last_cost is not None and abs(last_cost - result.cost) < 1e-8:
            break
        last_cost = result.cost

    s_values, pred, residuals = evaluate_curve(points, weights, params, s_values)
    _, e1, e2, _, kappa = unpack_params(params)
    arc_length = float(s_values[-1] - s_values[0]) if len(s_values) else 0.0
    model_type = "line" if abs(kappa) * arc_length < LINE_THRESHOLD_RAD else "arc"

    line_pred = line_center[None, :] + np.outer(line_s, line_dir)
    line_residuals = np.linalg.norm(line_pred - points, axis=1)
    line_cost = weighted_rms(line_residuals, weights)
    if model_type == "arc" and line_cost <= PREFER_LINE_RATIO * weighted_rms(residuals, weights):
        model_type = "line"

    if model_type == "line":
        e1 = line_dir
        e2 = orthogonal_vector(e1)
        n = normalize(np.cross(e1, e2))
        kappa = 0.0
        params = parameter_pack(line_center, e1, n, kappa)
        s_values, pred, residuals = evaluate_curve(points, weights, params, line_s)
        arc_length = float(s_values[-1] - s_values[0]) if len(s_values) else 0.0

    return {
        "params": params,
        "e1": e1,
        "e2": e2,
        "kappa": kappa,
        "s_values": s_values,
        "model_type": model_type,
        "arc_length": arc_length,
    }


def fit_constant_curvature(points_world):
    """Fit in a normalized frame (centered, scaled by the chord) and map back to world units.

    Returns the fields consumed by ``build_fitted_trajectory``: the in-plane axes
    e1, e2, kappa, arc_length, model_type, per-raw-point s_values and
    predicted_points, and the shape-fit quality label / relative error.
    """
    centroid = points_world.mean(axis=0)
    scale = float(np.linalg.norm(points_world[-1] - points_world[0]))
    scale = max(scale, 1e-3)
    points = (points_world - centroid[None, :]) / scale

    raw_weights = np.where(mark_suspicious_steps(points), 0.25, 1.0).astype(np.float64)
    window_radius = max(3, int(round(0.03 * len(points))))
    smoothed_points = robust_smooth_points(points, window_radius, raw_weights)
    arclength = compute_cumulative_arclength(smoothed_points)
    fit_points, fit_weights, targets = build_support_points(
        points, smoothed_points, arclength, raw_weights,
    )

    core = fit_constant_curvature_core(fit_points, fit_weights)

    raw_initial_s = None
    if len(targets) >= 2:
        raw_initial_s = np.interp(arclength, targets, core["s_values"])
    raw_s, raw_pred, raw_residuals = evaluate_curve(
        points, raw_weights, core["params"], initial_s=raw_initial_s,
    )
    _, _, support_residuals = evaluate_curve(
        fit_points, fit_weights, core["params"], initial_s=core["s_values"],
    )

    # Quality: max of support-point and raw-point weighted RMS, relative to arc length.
    arc_length = float(scale * core["arc_length"])
    support_rel = scale * weighted_rms(support_residuals, fit_weights) / max(arc_length, EPS)
    raw_rel = weighted_rms(scale * raw_residuals, raw_weights) / max(arc_length, EPS)
    relative_error = max(support_rel, raw_rel)
    quality = next((name for name, bound in QUALITY_THRESHOLDS if relative_error < bound), "poor")

    return {
        "e1": core["e1"].tolist(),
        "e2": core["e2"].tolist(),
        "kappa": float(core["kappa"] / scale),
        "s_values": (scale * raw_s).tolist(),
        "predicted_points": (centroid[None, :] + scale * raw_pred).tolist(),
        "model_type": core["model_type"],
        "arc_length": arc_length,
        "shape_fit_quality": quality,
        "shape_fit_relative_error": relative_error,
    }


def process_cam_dir(cam_dir):
    """Fit ``<cam_dir>/sam3_mask_traj.json`` and merge the result into ``<cam_dir>/trajectory.json``."""
    cam_dir = Path(cam_dir)
    traj_path = cam_dir / "sam3_mask_traj.json"
    data = json.loads(traj_path.read_text())
    # Fit only frames that have both a 3D and a 2D centroid.
    traj3d = {int(item["frame_idx"]): item["position_3d"] for item in data.get("trajectory_3d", [])}
    traj2d = {int(item["frame_idx"]) for item in data.get("trajectory_2d", [])}
    points_3d = np.asarray([traj3d[idx] for idx in sorted(set(traj3d) & traj2d)], dtype=np.float64)
    if len(points_3d) == 0:
        raise ValueError(f"Empty trajectory in {traj_path}")

    fit = fit_constant_curvature(points_3d)
    fitted = build_fitted_trajectory(
        fit,
        data.get("trajectory_3d", []),
        data.get("trajectory_2d", []),
        data["camera_info"]["intrinsics"],
    )
    save_to_trajectory_json(cam_dir, fitted)
    return fit
