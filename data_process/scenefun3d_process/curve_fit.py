"""Stage 4: motion curve targets for the sam-match-valid cams.

The trajectories are sampled from the annotated motion, so the constant-curvature fit is
analytic: a line along the motion axis (translation) or the circle of the swept part centroid
(rotation). curve_fit/convert_cc_to_fsframe.py turns it into fsparams and the cubic Bezier
spline_params. Each trajectories[] entry of trajectory.json gets fsparams, spline_params,
the fitted trajectory_3d / trajectory_2d (the sampled ones are kept as raw_trajectory_3d /
raw_trajectory_2d) and motion_3d / motion_2d recomputed from the fitted curve.
"""
import json
from pathlib import Path

import numpy as np

from curve_fit.convert_cc_to_fsframe import cc_fit_to_fsparams, integrate_fsframe, spline_ctrl_from_fsparams

from .frame_selection import SOURCE
from .sam_match import MANIFEST_NAME

FPS = 10.0  # the 15 samples are treated as 10 FPS: motion_3d / motion_2d end at sample 10
_EPS = 1e-9


def _normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < _EPS else v / n


def _orthogonal_unit(v) -> np.ndarray:
    """A unit vector orthogonal to v (fixed choice: it sets fsparams u, v for lines)."""
    v = _normalize(v)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(v, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    return _normalize(np.cross(v, basis))


def _fit(e1, e2, kappa: float, arc_length: float, model_type: str) -> dict:
    """Constant-curvature fit dict as read by cc_fit_to_fsparams (zero residual)."""
    return {"e1": e1, "e2": e2, "kappa": kappa, "s_values": [0.0], "arc_length": arc_length,
            "model_type": model_type, "shape_fit_quality": "perfect", "shape_fit_relative_error": 0.0}


def _line_fit(direction, distance_m: float, orient: str = "outwards") -> dict:
    if np.linalg.norm(direction) < _EPS:
        raise ValueError("direction must be non-zero")
    e1 = _normalize(direction) * (1.0 if orient == "outwards" else -1.0)
    return _fit(e1, _orthogonal_unit(e1), 0.0, float(distance_m), "line")


def analytic_cc_fit(params: dict, traj: np.ndarray) -> dict:
    """Fit for scenefun3d_motion_params; the curve starts at traj[0]."""
    if params["motion_type"] == "trans":
        return _line_fit(np.asarray(params["motion_dir_cam"]), params["distance_m"], params.get("orient", "outwards"))
    if params["motion_type"] != "rot":
        raise ValueError(f"unknown motion_type: {params['motion_type']!r}")
    axis = np.asarray(params["motion_dir_cam"])
    if np.linalg.norm(axis) < _EPS:
        raise ValueError("axis must be non-zero")
    axis = _normalize(axis)
    rel = traj[0] - np.asarray(params["origin_cam"], dtype=np.float64)
    r_vec = rel - np.dot(rel, axis) * axis
    radius = float(np.linalg.norm(r_vec))
    if radius < 1e-6:  # centroid on the axis: straight line through the samples
        delta = traj[1] - traj[0]
        dist = float(np.linalg.norm(delta))
        direction = np.array([1.0, 0.0, 0.0]) if dist < _EPS else delta / dist
        return _line_fit(direction, float(np.linalg.norm(traj[-1] - traj[0])))
    tangent0 = _normalize(np.cross(axis, r_vec))
    if np.dot(tangent0, traj[1] - traj[0]) < 0:  # match the sweep direction of the samples
        axis = -axis
        tangent0 = _normalize(np.cross(axis, r_vec))
    return _fit(tangent0, -_normalize(r_vec), 1.0 / radius, float(params["angle_rad"]) * radius, "arc")


def _project(points, intr: dict) -> list:
    """Camera-frame points -> pixels; [0, 0] for z <= 0."""
    pts = np.asarray(points, dtype=np.float64)
    z = pts[:, 2]
    valid = z > _EPS
    safe_z = np.where(valid, z, 1.0)
    u = np.where(valid, float(intr["fx"]) * pts[:, 0] / safe_z + float(intr["cx"]), 0.0)
    v = np.where(valid, float(intr["fy"]) * pts[:, 1] / safe_z + float(intr["cy"]), 0.0)
    return np.stack([u, v], axis=1).tolist()


def fit_entry(entry: dict, intr: dict) -> None:
    """Add the fitted fields to one trajectories[] entry (idempotent: refits from raw_trajectory_3d)."""
    raw = entry.get("raw_trajectory_3d") or entry["trajectory_3d"]
    raw_3d = np.asarray([p["position_3d"] for p in raw], dtype=np.float64)
    fsp = cc_fit_to_fsparams(analytic_cc_fit(entry["scenefun3d_motion_params"], raw_3d), raw)
    spline = spline_ctrl_from_fsparams(fsp)
    fitted_3d = integrate_fsframe(fsp, num_samples=len(raw) - 1)  # len(raw) points
    fitted_2d = _project(fitted_3d, intr)
    end = min(max(1, int(round(FPS))), len(fitted_3d) - 1)
    disp = fitted_3d[end] - fitted_3d[0]
    norm = float(np.linalg.norm(disp))

    entry.setdefault("raw_trajectory_3d", entry["trajectory_3d"])
    entry["raw_trajectory_2d"] = [{"frame_idx": p["frame_idx"], "position_pixel": px}
                                  for p, px in zip(raw, _project(raw_3d, intr))]
    entry["trajectory_3d"] = [{"frame_idx": p["frame_idx"], "position_3d": fitted_3d[i].tolist()}
                              for i, p in enumerate(raw)]
    entry["trajectory_2d"] = [{"frame_idx": p["frame_idx"], "position_pixel": fitted_2d[i]}
                              for i, p in enumerate(raw)]
    entry["motion_3d"] = (disp / norm).tolist() if norm > 1e-10 else [0.0, 0.0, 0.0]
    entry["motion_2d"] = {"start_pixel": [float(x) for x in fitted_2d[0]],
                          "end_pixel": [float(x) for x in fitted_2d[end]]}
    entry["fsparams"] = fsp
    entry["spline_params"] = spline


def fit_cam(cam_dir: Path) -> int:
    """Rewrite cam_dir/trajectory.json with the fitted fields; returns the number of trajectories."""
    path = cam_dir / "trajectory.json"
    tj = json.loads(path.read_text())
    tj["camera_info"].setdefault("distortion", None)
    for entry in tj["trajectories"]:
        fit_entry(entry, tj["camera_info"]["intrinsics"])
    path.write_text(json.dumps(tj, indent=2))
    return len(tj["trajectories"])


def run_curve_fit(output_root: Path) -> None:
    sam_match = json.loads((output_root / MANIFEST_NAME).read_text())
    n_cams = n_traj = 0
    for key, verdict in sam_match.items():
        if verdict.get("status") != "valid":
            continue
        episode, interval_idx, cam = key.split("__")
        n_traj += fit_cam(output_root / SOURCE / episode / f"interval_{interval_idx}" / cam)
        n_cams += 1
    print(f"[curve-fit] {n_cams} cams, {n_traj} trajectories")
