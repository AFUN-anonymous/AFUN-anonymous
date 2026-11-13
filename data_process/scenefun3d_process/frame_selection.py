"""Stage 1: pick frames per task description and write candidate cams.

For each video of a visit and each description (task instruction) with annotated motions:
  * frames (10 FPS hires stream) where the motion origin projects inside the image, in front
    of the camera, and agrees with the rendered depth within DEPTH_PRE_FILTER_M are candidates;
  * 20-60 evenly spaced candidates are kept per annotation (about one per 10 frames);
  * the annotated motion is sampled directly as a 3D curve (translation: 0.3 m along the axis;
    rotation: 90 degree sweep of the part centroid about the axis) and moved to the camera frame.

Writes {output_root}/scenefun3d/scenefun3d_<visit>_<video>/interval_<desc idx>/cam_<k>/ with
obs_frame.png, obs_frame_depth.npy (float32 mm) and trajectory.json ({camera_info, trajectories:
one entry per annotation of the description visible in that frame}), plus metadata.json per
interval and episode_summary.json per episode. Per-visit manifests go to {output_root}/manifests/
and are merged into {output_root}/frame_selection_manifest.json, alignment_report.json and
skipped_videos.json.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .data import (
    load_annotations, load_depth_raw, load_descriptions, load_frame_color, load_frame_depth,
    load_intrinsics, load_laser_scan, load_motions, load_poses,
)

SOURCE = "scenefun3d"
ALIGN_MAX_PROBES = 10            # timestamps tried per video to resolve pose direction / depth unit
ALIGN_ACCEPT_DEPTH_ERR_M = 0.1   # accepted back-projection error of the motion origin
DEPTH_PRE_FILTER_M = 0.5         # |depth at origin pixel - origin z| above this -> occluded frame
TARGET_STRIDE, MIN_CAMS, MAX_CAMS = 10, 20, 60
N_SAMPLES = 15                   # points per sampled motion curve
TRANS_DISTANCE_M = 0.3
ROT_ANGLE_DEG = 90.0
FPS = 10.0
CAM_SOURCES = {
    "extrinsic_source": "scenefun3d_verify_alignment",
    "intrinsic_source": "scenefun3d_pincam",
    "depth_source": "hires_depth_faro",
}
MANIFEST_NAMES = {
    "frame_selection": "frame_selection_manifest.json",
    "alignment_report": "alignment_report.json",
    "skipped_videos": "skipped_videos.json",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _w2c(poses: dict, ts: str, pose_direction: str):
    """World->camera (R, t) at ts for the resolved pose direction."""
    R, t = poses[ts]
    if pose_direction == "w2c":
        return R, t
    return R.T, -R.T @ t


def _project(points_w, R_w2c, t_w2c, intr: dict):
    """World points (N, 3) -> pixels (N, 2) and camera z (N,)."""
    X_c = (R_w2c @ np.asarray(points_w, dtype=np.float64).reshape(-1, 3).T).T + t_w2c
    z = X_c[:, 2]
    z_safe = np.where(z == 0, 1e-12, z)
    u = intr["fx"] * X_c[:, 0] / z_safe + intr["cx"]
    v = intr["fy"] * X_c[:, 1] / z_safe + intr["cy"]
    return np.stack([u, v], axis=1), z


def _project_cam(p, intr: dict):
    """Camera-frame point -> (u, v), or None when behind the camera."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if z <= 0:
        return None
    return (float(intr["fx"]) * x / z + float(intr["cx"]), float(intr["fy"]) * y / z + float(intr["cy"]))


def _sample_trans(origin, direction, distance_m: float, n: int, orient: str) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    sign = 1.0 if orient == "outwards" else -1.0
    t = np.linspace(0.0, sign * distance_m, n)
    return np.asarray(origin, dtype=np.float64)[None, :] + t[:, None] * direction[None, :]


def _sample_rot(axis_origin, axis_dir, ref_point, angle_deg: float, n: int) -> np.ndarray:
    """Rotate ref_point about the axis line by angles linspace(0, angle_deg, n) (right-hand rule)."""
    axis_origin = np.asarray(axis_origin, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    p = np.asarray(ref_point, dtype=np.float64) - axis_origin
    K = np.array([[0.0, -axis_dir[2], axis_dir[1]],
                  [axis_dir[2], 0.0, -axis_dir[0]],
                  [-axis_dir[1], axis_dir[0], 0.0]])
    K2 = K @ K
    out = np.empty((n, 3))
    for i, a in enumerate(np.deg2rad(np.linspace(0.0, angle_deg, n))):
        out[i] = (np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * K2) @ p + axis_origin
    return out


def trajectory_camera_frame(scan, motion: dict, annotation: dict, R_w2c, t_w2c) -> np.ndarray:
    """Sample the annotated motion in world frame -> (N_SAMPLES, 3) camera-frame points."""
    origin_w = np.asarray(scan[motion["motion_origin_idx"]], dtype=np.float64)
    direction = np.asarray(motion["motion_dir"], dtype=np.float64)
    if motion["motion_type"] == "trans":
        pts_w = _sample_trans(origin_w, direction, TRANS_DISTANCE_M, N_SAMPLES,
                              motion.get("motion_viz_orient", "outwards"))
    elif motion["motion_type"] == "rot":
        ref = scan[np.asarray(annotation["indices"], dtype=np.int64)].mean(axis=0)
        pts_w = _sample_rot(origin_w, direction, ref, ROT_ANGLE_DEG, N_SAMPLES)
    else:
        raise ValueError(f"unknown motion_type: {motion['motion_type']!r}")
    return (np.asarray(R_w2c) @ np.asarray(pts_w).T).T + np.asarray(t_w2c)


def motion_params_camera_frame(scan, motion: dict, annotation: dict, R_w2c, t_w2c) -> dict:
    """Camera-frame motion annotation, read by curve_fit (trans: axis + distance; rot: axis + pivot + ref)."""
    R_w2c = np.asarray(R_w2c, dtype=np.float64)
    t_w2c = np.asarray(t_w2c, dtype=np.float64)
    origin_cam = (R_w2c @ np.asarray(scan[motion["motion_origin_idx"]], dtype=np.float64)) + t_w2c
    dir_cam = R_w2c @ np.asarray(motion["motion_dir"], dtype=np.float64)
    if motion["motion_type"] == "trans":
        return {"motion_type": "trans", "motion_dir_cam": dir_cam.tolist(), "origin_cam": origin_cam.tolist(),
                "distance_m": TRANS_DISTANCE_M, "orient": motion.get("motion_viz_orient", "outwards")}
    if motion["motion_type"] == "rot":
        ref_cam = (R_w2c @ scan[np.asarray(annotation["indices"], dtype=np.int64)].mean(axis=0)) + t_w2c
        return {"motion_type": "rot", "motion_dir_cam": dir_cam.tolist(), "origin_cam": origin_cam.tolist(),
                "ref_cam": ref_cam.tolist(), "angle_rad": float(np.deg2rad(ROT_ANGLE_DEG))}
    raise ValueError(f"unknown motion_type: {motion['motion_type']!r}")


# ---------------------------------------------------------------------------
# Alignment (pose direction and depth unit are not documented; resolve them per video)
# ---------------------------------------------------------------------------

def _probe_alignment(scan, motions, anns, video_dir, video_id, poses, ts):
    """Best (depth_err_m, pose_direction, depth_unit) over all motion origins at ts, or None.

    The error is the distance between a motion origin and the back-projection of the depth
    pixel it projects to (5x5 median when that pixel is a hole); inf without depth.
    """
    intr = load_intrinsics(video_dir, video_id, ts)
    depth_raw = load_depth_raw(video_dir, video_id, ts)
    R, t = poses[ts]
    candidates = {"c2w": (R.T, -R.T @ t), "w2c": (R, t)}
    best = None
    for m in motions:
        if m["annot_id"] not in anns:
            continue
        origin_w = scan[m["motion_origin_idx"]]
        for name, (R_w2c, t_w2c) in candidates.items():
            uv, z = _project(origin_w[None, :], R_w2c, t_w2c, intr)
            uv, z = uv[0], z[0]
            if z <= 0 or not (0 <= uv[0] < intr["w"] and 0 <= uv[1] < intr["h"]):
                continue
            u_px = max(0, min(intr["w"] - 1, int(round(uv[0]))))
            v_px = max(0, min(intr["h"] - 1, int(round(uv[1]))))
            d_raw = float(depth_raw[v_px, u_px])
            if d_raw == 0:
                win = depth_raw[max(0, v_px - 2):v_px + 3, max(0, u_px - 2):u_px + 3].astype(np.float32)
                win = win[win > 0]
                d_raw = float(np.median(win)) if win.size else 0.0
            for unit, unit_div in (("mm", 1000.0), ("m", 1.0)):
                if d_raw == 0:
                    err = np.inf
                else:
                    z_m = d_raw / unit_div
                    X_c = np.array([(uv[0] - intr["cx"]) * z_m / intr["fx"],
                                    (uv[1] - intr["cy"]) * z_m / intr["fy"], z_m])
                    err = float(np.linalg.norm(R_w2c.T @ X_c + (-R_w2c.T @ t_w2c) - origin_w))
                if best is None or err < best[0]:
                    best = (err, name, unit)
    return best


def resolve_alignment(scan, motions, anns, video_dir, video_id, poses, all_ts) -> dict:
    """Try ALIGN_MAX_PROBES evenly spaced timestamps; first one with error <= threshold wins."""
    probe_idx = np.linspace(0, len(all_ts) - 1, num=min(ALIGN_MAX_PROBES, len(all_ts)), dtype=int)
    for attempt, idx in enumerate(probe_idx, start=1):
        ts = all_ts[int(idx)]
        best = _probe_alignment(scan, motions, anns, video_dir, video_id, poses, ts)
        if best is not None and best[0] <= ALIGN_ACCEPT_DEPTH_ERR_M:
            return {"pose_direction": best[1], "depth_unit": best[2], "depth_err_m": best[0],
                    "probe_timestamp": ts, "probe_attempts": attempt}
    raise RuntimeError(f"alignment failed across {len(probe_idx)} probes")


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------

def compute_num_cams(W: int) -> int:
    """min(W, clamp(W // TARGET_STRIDE, MIN_CAMS, MAX_CAMS))."""
    if W <= 0:
        return 0
    return min(W, min(MAX_CAMS, max(MIN_CAMS, W // TARGET_STRIDE)))


def pick_frame_indices(W: int, num_cams: int) -> list:
    """num_cams evenly spaced, de-duplicated indices in [0, W)."""
    if W == 0 or num_cams == 0:
        return []
    if num_cams == 1:
        return [0]
    out = []
    for i in np.linspace(0, W - 1, num=num_cams).round().astype(int):
        if int(i) not in out:
            out.append(int(i))
    return out


def select_candidate_timestamps(origin_w, intr: dict, poses: dict, pose_direction: str, depth_at) -> list:
    """Timestamps where the origin is in front of the camera, inside the image, and not occluded.

    depth_at(ts, u, v) returns meters; 0 (unknown) never filters.
    """
    origin_w = np.asarray(origin_w, dtype=np.float64).reshape(3)
    candidates = []
    for ts in sorted(poses, key=float):
        R_w2c, t_w2c = _w2c(poses, ts, pose_direction)
        uv, z = _project(origin_w[None, :], R_w2c, t_w2c, intr)
        u, v, z_cam = float(uv[0, 0]), float(uv[0, 1]), float(z[0])
        if z_cam <= 0 or not (0 <= u < intr["w"] and 0 <= v < intr["h"]):
            continue
        d = depth_at(ts, u, v)
        if d > 0 and abs(d - z_cam) > DEPTH_PRE_FILTER_M:
            continue
        candidates.append(ts)
    return candidates


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    _write_json(tmp, manifest)
    tmp.replace(path)


def _write_cam(cam_dir: Path, row: dict, trajectories: list) -> None:
    cam_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(cam_dir / "obs_frame.png"), row["rgb"]):
        raise RuntimeError(f"cv2.imwrite failed: {cam_dir / 'obs_frame.png'}")
    np.save(cam_dir / "obs_frame_depth.npy", (row["depth"] * 1000.0).astype(np.float32))  # m -> mm
    T = np.eye(4)
    T[:3, :3] = np.asarray(row["R_w2c"])
    T[:3, 3] = np.asarray(row["t_w2c"])
    intr = row["intrinsics"]
    camera_info = {
        "cam_id": row["cam_id"],
        "intrinsics": {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")},
        "T_base_to_cam": T.tolist(),
        **CAM_SOURCES,
    }
    _write_json(cam_dir / "trajectory.json", {"camera_info": camera_info, "trajectories": trajectories})


def _trajectory_entry(row: dict, member: dict, intr: dict):
    """One trajectories[] entry; None when either curve endpoint is behind the camera."""
    traj = row["trajectory_3d"]
    p_start = np.asarray(traj[0]["position_3d"], dtype=np.float64)
    p_end = np.asarray(traj[-1]["position_3d"], dtype=np.float64)
    delta = p_end - p_start
    n = float(np.linalg.norm(delta))
    uv_start, uv_end = _project_cam(p_start, intr), _project_cam(p_end, intr)
    if uv_start is None or uv_end is None:
        return None
    return {
        "annot_id": member["annot_id"],
        "interval_language": member["interval_language"],
        "motion_type": member["motion_type"],
        "trajectory_3d": traj,
        "motion_3d": (delta / n).tolist() if n > 1e-9 else [0.0, 0.0, 0.0],
        "motion_2d": {"start_pixel": [float(uv_start[0]), float(uv_start[1])],
                      "end_pixel": [float(uv_end[0]), float(uv_end[1])]},
        "scenefun3d_motion_params": row["motion_params"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _process_annotation(spec, ctx):
    """Sample frames for one (description, annotation); returns the per-frame rows or None."""
    desc_idx, desc, annot_id, motion, annotation = spec
    scan, poses, video_dir, video_id = ctx["scan"], ctx["poses"], ctx["video_dir"], ctx["video_id"]
    depth_cache = {}

    def depth_at(ts, u, v):
        if ts not in depth_cache:
            try:
                depth_cache[ts] = load_frame_depth(video_dir, video_id, ts)
            except FileNotFoundError:
                depth_cache[ts] = None
        d = depth_cache[ts]
        if d is None:
            return 0.0
        u_i, v_i = int(round(u)), int(round(v))
        if 0 <= u_i < d.shape[1] and 0 <= v_i < d.shape[0]:
            return float(d[v_i, u_i])
        return 0.0

    try:
        intr_any = load_intrinsics(video_dir, video_id, ctx["all_ts"][0])
    except FileNotFoundError:
        return None
    candidates = select_candidate_timestamps(scan[motion["motion_origin_idx"]], intr_any, poses,
                                             ctx["pose_direction"], depth_at)
    chosen_ts = [candidates[i] for i in pick_frame_indices(len(candidates), compute_num_cams(len(candidates)))]
    if not chosen_ts:
        return None
    frame_indices = [ctx["frame_index_by_ts"][ts] for ts in chosen_ts]

    rows = []
    for cam_idx, ts in enumerate(chosen_ts):
        try:
            rgb = load_frame_color(video_dir, video_id, ts)
            depth = load_frame_depth(video_dir, video_id, ts)
            intr = load_intrinsics(video_dir, video_id, ts)
        except FileNotFoundError as e:
            print(f"[WARN] missing asset {e}; skipping cam {cam_idx}")
            continue
        R_w2c, t_w2c = _w2c(poses, ts, ctx["pose_direction"])
        traj = trajectory_camera_frame(scan, motion, annotation, R_w2c, t_w2c)
        rows.append({
            "cam_slot": f"cam_{cam_idx}", "cam_id": video_id, "frame_index": ctx["frame_index_by_ts"][ts],
            "timestamp": ts, "rgb": rgb, "depth": depth, "intrinsics": intr,
            "R_w2c": np.asarray(R_w2c), "t_w2c": np.asarray(t_w2c),
            "trajectory_3d": [{"frame_idx": int(i), "position_3d": [float(x), float(y), float(z)]}
                              for i, (x, y, z) in enumerate(np.asarray(traj))],
            "motion_params": motion_params_camera_frame(scan, motion, annotation, R_w2c, t_w2c),
        })
    if not rows:
        return None
    return {"desc_idx": desc_idx, "annot_id": annot_id, "desc_id": desc["desc_id"],
            "interval_language": desc["description"], "motion_type": motion["motion_type"],
            "start_frame": min(frame_indices), "end_frame": max(frame_indices), "rows": rows}


def _write_description(members: list, desc_idx: int, ep_dir: Path, episode_id: str,
                       visit_id: str, video_id: str, manifest: dict):
    """Merge the annotations of one description per frame into cams; returns the interval summary."""
    members = sorted(members, key=lambda r: r["annot_id"])
    interval_idx = f"{desc_idx:03d}"
    interval_dir = ep_dir / f"interval_{interval_idx}"
    frame_map = {}
    for member in members:
        for row in member["rows"]:
            frame_map.setdefault((row["cam_id"], row["frame_index"]), []).append((member, row))

    source_annot_ids = [m["annot_id"] for m in members]
    first = members[0]
    meta = {
        "episode_id": episode_id,
        "interval_idx": interval_idx,
        "source_annot_ids": source_annot_ids,
        "desc_id": first["desc_id"],
        "interval_language": first["interval_language"],
        "motion_type": first["motion_type"],
        "start_frame": min(m["start_frame"] for m in members),
        "end_frame": max(m["end_frame"] for m in members),
        "num_annotations": len(members),
    }
    out_slot = 0
    for key in sorted(frame_map, key=lambda k: (k[1], str(k[0]))):
        matches = sorted(frame_map[key], key=lambda pair: pair[0]["annot_id"])
        first_row = matches[0][1]
        trajectories, provenance = [], []
        for member, row in matches:
            entry = _trajectory_entry(row, member, first_row["intrinsics"])
            if entry is None:
                continue
            trajectories.append(entry)
            provenance.append({"annot_id": member["annot_id"], "cam_slot": row["cam_slot"]})
        if not trajectories:
            continue
        _write_cam(interval_dir / f"cam_{out_slot}", first_row, trajectories)
        meta[f"cam_{out_slot}"] = {
            "cam_id": first_row["cam_id"], "frame_index": first_row["frame_index"],
            "timestamp": first_row["timestamp"], **CAM_SOURCES,
            "K_trajectories": len(trajectories), "provenance": provenance,
        }
        manifest[f"{episode_id}__{interval_idx}__cam_{out_slot}"] = {
            "status": "pending_mask", "visit_id": visit_id, "video_id": video_id,
            "timestamp": first_row["timestamp"], "interval_idx": interval_idx,
            "motion_type": first["motion_type"], "language": first["interval_language"],
        }
        out_slot += 1
    if out_slot == 0:
        return None
    meta["num_cams"] = out_slot
    _write_json(interval_dir / "metadata.json", meta)
    return {"idx": interval_idx, "start": meta["start_frame"], "end": meta["end_frame"],
            "length": meta["end_frame"] - meta["start_frame"] + 1, "num_cams": out_slot,
            "source_annot_ids": source_annot_ids, "num_annotations": len(members),
            "desc_id": first["desc_id"], "motion_type": first["motion_type"]}


def process_visit(dataset_root: Path, output_root: Path, visit_id: str) -> None:
    """Write all cams of one visit and its per-visit manifests (skipped if already done)."""
    manifests_dir = output_root / "manifests"
    fsm_path = manifests_dir / f"frame_selection_{visit_id}.json"
    if fsm_path.exists():
        print(f"[skip] {visit_id}: {fsm_path.name} already present")
        return
    visit_dir = dataset_root / "train_val" / visit_id
    if not visit_dir.exists():
        visit_dir = dataset_root / "test" / visit_id
    try:
        scan = load_laser_scan(visit_dir, visit_id)
        motions = load_motions(visit_dir, visit_id)
        anns = {a["annot_id"]: a for a in load_annotations(visit_dir, visit_id)}
        descriptions = load_descriptions(visit_dir, visit_id)
    except FileNotFoundError as e:
        print(f"[WARN] visit {visit_id}: {e}; skipping")
        return
    motion_by_annot = {}
    for m in motions:
        motion_by_annot.setdefault(m["annot_id"], m)

    manifest, align_report, skipped = {}, {}, {}
    for video_id in sorted(d.name for d in visit_dir.iterdir() if d.is_dir() and d.name.isdigit()):
        video_dir = visit_dir / video_id
        poses = load_poses(video_dir)
        all_ts = sorted(poses, key=float)
        try:
            aln = resolve_alignment(scan, motions, anns, video_dir, video_id, poses, all_ts)
        except RuntimeError as e:
            print(f"[SKIP] {visit_id}/{video_id}: {e}")
            skipped[f"{visit_id}/{video_id}"] = {"reason": str(e)}
            continue
        align_report[f"{visit_id}/{video_id}"] = aln

        ctx = {"scan": scan, "poses": poses, "all_ts": all_ts, "video_dir": video_dir, "video_id": video_id,
               "pose_direction": aln["pose_direction"],
               "frame_index_by_ts": {ts: i for i, ts in enumerate(all_ts)}}
        specs = [(desc_idx, desc, annot_id, motion_by_annot[annot_id], anns[annot_id])
                 for desc_idx, desc in enumerate(descriptions)
                 for annot_id in desc.get("annot_id", [])
                 if annot_id in anns and annot_id in motion_by_annot]
        with ThreadPoolExecutor(max(1, min(8, os.cpu_count() or 1))) as ex:
            results = list(ex.map(lambda s: _process_annotation(s, ctx), specs))

        by_desc = {}
        for r in results:
            if r is not None:
                by_desc.setdefault(r["desc_idx"], []).append(r)
        episode_id = f"{SOURCE}_{visit_id}_{video_id}"
        ep_dir = output_root / SOURCE / episode_id
        intervals = []
        for desc_idx in sorted(by_desc):
            summary = _write_description(by_desc[desc_idx], desc_idx, ep_dir, episode_id,
                                         visit_id, video_id, manifest)
            if summary is not None:
                intervals.append(summary)
        if intervals:
            _write_json(ep_dir / "episode_summary.json", {
                "episode_id": episode_id, "episode_path": f"train_val/{visit_id}/{video_id}",
                "num_frames": len(all_ts), "num_intervals": len(intervals),
                "num_cams": sum(iv["num_cams"] for iv in intervals), "fps": FPS, "intervals": intervals,
            })

    manifests_dir.mkdir(parents=True, exist_ok=True)
    _save_manifest(manifests_dir / f"alignment_report_{visit_id}.json", align_report)
    _save_manifest(manifests_dir / f"skipped_videos_{visit_id}.json", skipped)
    _save_manifest(fsm_path, manifest)
    print(f"[done] {visit_id}: {len(manifest)} cams, {len(skipped)} videos skipped")


def merge_manifests(output_root: Path) -> None:
    """Merge {output_root}/manifests/<prefix>_<visit>.json into the root-level manifests."""
    for prefix, name in MANIFEST_NAMES.items():
        merged = {}
        files = sorted((output_root / "manifests").glob(f"{prefix}_*.json"))
        for path in files:
            merged.update(json.loads(path.read_text()))
        _save_manifest(output_root / name, merged)
        print(f"  {name}: {len(merged)} entries from {len(files)} visit(s)")


def run_frame_selection(dataset_root: Path, output_root: Path, visits: list) -> None:
    for visit_id in visits:
        process_visit(dataset_root, output_root, visit_id)
    merge_manifests(output_root)
