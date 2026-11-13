#!/usr/bin/env python3
"""Convert a constant-curvature fit into the stored motion targets.

``fsparams``: Frenet-Serret frame parameters of the fitted arc, anchored at the
first raw trajectory point r0 (initial tangent T0, normal hint (u, v), curvature,
zero torsion, velocity = arc length).

``spline_params``: the training target. The FS-frame arc is integrated densely
from r0 and a cubic Bezier is fit to it by least squares on the two interior
control points, with both endpoints fixed. Stored as
``{"ctrl": [P1 - P0, P2 - P0, P3 - P0]}``, i.e. relative to the anchor P0 = r0.

``save_to_trajectory_json`` merges both into the cam's ``trajectory.json``,
keeping the previous ``trajectory_3d`` / ``trajectory_2d`` as ``raw_trajectory_*``
and replacing them with the fitted curve.
"""
import json
import math
from pathlib import Path

import numpy as np

EPS = 1e-8
BEZIER_SAMPLES = 200  # integration steps along the FS-frame arc before the Bezier fit


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        return v * 0.0
    return v / n


def _trajectory_linearity_ratio(points: np.ndarray) -> float:
    """Max perpendicular deviation from chord / chord length."""
    if len(points) < 2:
        return 0.0
    chord = points[-1] - points[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len < EPS:
        polyline_len = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        return polyline_len / EPS
    T = chord / chord_len
    rel = points - points[0]
    proj = np.outer(rel @ T, T)
    deviations = np.linalg.norm(rel - proj, axis=1)
    return float(np.max(deviations)) / chord_len


def _invert_initial_frame(T: np.ndarray, N: np.ndarray) -> tuple:
    """Inverse of ``_make_initial_frame``: find (T0, u, v) such that projecting
    [u, v, 0] onto the plane orthogonal to T gives N."""
    T = _normalize(T)
    Tz = float(T[2])
    if abs(Tz) > EPS:
        alpha = -float(N[2]) / Tz
        n_raw = N + alpha * T
        u, v = float(n_raw[0]), float(n_raw[1])
    else:
        u, v = float(N[0]), float(N[1])
    return T, u, v


def cc_fit_to_fsparams(fit: dict, trajectory_3d: list) -> dict:
    """Constant-curvature fit -> FS-frame parameters anchored at the first raw point.

    Returns a dict with keys r0, T0, u, v, kappa, tau, velocity, fit_mode,
    linearity_ratio, shape_fit_quality, shape_fit_relative_error.
    """
    e1 = np.asarray(fit["e1"], dtype=np.float64)
    e2 = np.asarray(fit["e2"], dtype=np.float64)
    kappa_cc = float(fit["kappa"])
    s_values = fit["s_values"]
    arc_length = float(fit["arc_length"])
    model_type = fit["model_type"]

    points_3d = np.asarray(
        [p["position_3d"] for p in trajectory_3d], dtype=np.float64
    )
    r0 = points_3d[0].tolist()

    s0 = float(s_values[0]) if len(s_values) else 0.0
    ks0 = kappa_cc * s0

    # Tangent at s0
    T_start = math.cos(ks0) * e1 + math.sin(ks0) * e2
    T_start = _normalize(T_start)

    # Normal and binormal at s0 (Frenet convention: N toward center)
    if kappa_cc >= 0:
        N_start = -math.sin(ks0) * e1 + math.cos(ks0) * e2
        kappa_fs = kappa_cc
    else:
        N_start = math.sin(ks0) * e1 - math.cos(ks0) * e2
        kappa_fs = -kappa_cc

    N_start = _normalize(N_start)

    T0, u, v = _invert_initial_frame(T_start, N_start)

    result = {
        "r0": r0,
        "T0": T0.tolist(),
        "u": u,
        "v": v,
        "kappa": float(kappa_fs),
        "tau": 0.0,
        "velocity": float(arc_length),
        "fit_mode": "line_aware" if model_type == "line" else "circle_fit",
        "linearity_ratio": float(_trajectory_linearity_ratio(points_3d)),
    }
    if "shape_fit_quality" in fit:
        result["shape_fit_quality"] = fit["shape_fit_quality"]
    if "shape_fit_relative_error" in fit:
        result["shape_fit_relative_error"] = float(fit["shape_fit_relative_error"])
    return result


def _fit_bezier_to_arc(arc_points: np.ndarray) -> list:
    """Least-squares cubic Bezier through dense arc points with P0, P3 fixed.

    t_i is the cumulative arc-length fraction. Returns [P1-P0, P2-P0, P3-P0].
    """
    P0 = arc_points[0]
    P3 = arc_points[-1]
    chord = P3 - P0

    if len(arc_points) < 3 or np.linalg.norm(chord) < EPS:
        return [
            (chord / 3.0).tolist(),
            (2.0 * chord / 3.0).tolist(),
            chord.tolist(),
        ]

    diffs = np.linalg.norm(np.diff(arc_points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = cum[-1]
    if total < EPS:
        return [
            (chord / 3.0).tolist(),
            (2.0 * chord / 3.0).tolist(),
            chord.tolist(),
        ]
    t = cum / total

    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    c1 = 3.0 * (1.0 - t) ** 2 * t
    c2 = 3.0 * (1.0 - t) * t ** 2
    rhs = arc_points - ((1.0 - t) ** 3)[:, None] * P0 - (t ** 3)[:, None] * P3

    A = np.column_stack([c1, c2])
    params, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
    P1, P2 = params[0], params[1]

    return [
        (P1 - P0).tolist(),
        (P2 - P0).tolist(),
        chord.tolist(),
    ]


def spline_ctrl_from_fsparams(fsparams: dict, num_samples: int = BEZIER_SAMPLES) -> dict:
    """Bezier control points (relative to fsparams["r0"]) of the integrated FS-frame arc."""
    velocity = float(fsparams.get("velocity", 0))
    if velocity < EPS:
        return {"ctrl": [[0.0, 0.0, 0.0]] * 3}

    arc_points = integrate_fsframe(fsparams, num_samples=num_samples)
    return {"ctrl": _fit_bezier_to_arc(arc_points)}


# ---------------------------------------------------------------------------
# FS frame integration
# ---------------------------------------------------------------------------

def _hat(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ], dtype=np.float64)


def _so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues: exponential map so(3) -> SO(3)."""
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3) + _hat(w)
    K = _hat(w / theta)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def _make_initial_frame(T0: np.ndarray, u: float, v: float) -> np.ndarray:
    """T0, u, v -> R0 = [T, N, B] (3x3)."""
    T = _normalize(T0)
    n_raw = np.array([u, v, 0.0], dtype=np.float64)
    n_proj = n_raw - np.dot(n_raw, T) * T
    if np.linalg.norm(n_proj) < EPS:
        a = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, T)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        n_proj = a - np.dot(a, T) * T
    N = _normalize(n_proj)
    B = np.cross(T, N)
    return np.column_stack([T, N, B])


def integrate_fsframe(fsparams: dict, num_samples: int = BEZIER_SAMPLES) -> np.ndarray:
    """Integrate the FS-frame curve over unit time; returns (num_samples + 1, 3) points.

    velocity = arc length, so the curve covers the full fitted arc.
    """
    r0 = np.asarray(fsparams["r0"], dtype=np.float64)
    T0 = np.asarray(fsparams["T0"], dtype=np.float64)
    kappa = float(fsparams["kappa"])
    tau = float(fsparams["tau"])
    velocity = float(fsparams["velocity"])
    u, v = float(fsparams["u"]), float(fsparams["v"])

    R = _make_initial_frame(T0, u, v)
    dt = 1.0 / float(num_samples)

    points = [r0.copy()]
    for _ in range(num_samples):
        ds = velocity * dt
        w_body = np.array([tau, 0.0, kappa], dtype=np.float64)
        T_dir = R[:, 0]
        r0 = r0 + ds * T_dir
        R = R @ _so3_exp(w_body * ds)
        points.append(r0.copy())
    return np.array(points, dtype=np.float64)


# ---------------------------------------------------------------------------
# trajectory.json
# ---------------------------------------------------------------------------

def _project_point(point_3d, intrinsics):
    x, y, z = float(point_3d[0]), float(point_3d[1]), float(point_3d[2])
    if z <= EPS:
        return None
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    return np.array([fx * x / z + cx, fy * y / z + cy], dtype=np.float64)


def build_fitted_trajectory(
    fit: dict,
    raw_trajectory_3d: list,
    raw_trajectory_2d: list,
    intrinsics: dict,
) -> dict:
    """Fitted trajectory_3d / trajectory_2d plus fsparams and spline_params."""
    fsparams = cc_fit_to_fsparams(fit, raw_trajectory_3d)
    spline_params = spline_ctrl_from_fsparams(fsparams)

    predicted_points = fit.get("predicted_points", [])
    frame_indices = [p["frame_idx"] for p in raw_trajectory_3d]
    fitted_traj_3d = [
        {"frame_idx": idx, "position_3d": pt}
        for idx, pt in zip(frame_indices, predicted_points)
    ]

    fitted_traj_2d = []
    for idx, pt in zip(frame_indices, predicted_points):
        px = _project_point(pt, intrinsics)
        if px is not None:
            fitted_traj_2d.append({"frame_idx": idx, "position_pixel": px.tolist()})
        else:
            # Fall back to raw 2D if projection fails
            raw_entry = next((r for r in raw_trajectory_2d if r["frame_idx"] == idx), None)
            if raw_entry:
                fitted_traj_2d.append(raw_entry)

    return {
        "trajectory_3d": fitted_traj_3d,
        "trajectory_2d": fitted_traj_2d,
        "fsparams": fsparams,
        "spline_params": spline_params,
    }


def save_to_trajectory_json(cam_dir: Path, fitted: dict) -> Path:
    """Merge fitted data into ``<cam_dir>/trajectory.json``.

    Keeps existing fields (camera_info, motion_2d, ...). On the first fit the
    existing trajectory_3d / trajectory_2d are kept as raw_trajectory_3d /
    raw_trajectory_2d; refits leave those untouched.
    """
    traj_path = Path(cam_dir) / "trajectory.json"
    if traj_path.exists():
        traj_data = json.loads(traj_path.read_text())
    else:
        traj_data = {}

    if "raw_trajectory_3d" not in traj_data and "trajectory_3d" in traj_data:
        traj_data["raw_trajectory_3d"] = traj_data["trajectory_3d"]
    if "raw_trajectory_2d" not in traj_data and "trajectory_2d" in traj_data:
        traj_data["raw_trajectory_2d"] = traj_data["trajectory_2d"]

    traj_data["trajectory_3d"] = fitted["trajectory_3d"]
    traj_data["trajectory_2d"] = fitted["trajectory_2d"]
    traj_data["fsparams"] = fitted["fsparams"]
    traj_data["spline_params"] = fitted["spline_params"]

    traj_path.write_text(json.dumps(traj_data, indent=2))
    return traj_path
