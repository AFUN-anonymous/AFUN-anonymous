"""HOI4D processing utilities -- parsing, extraction, projection."""

import csv
import json
import cv2
import numpy as np
from PIL import Image as PILImage
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    FPS, IMAGE_H, IMAGE_W, MANIPULATION_START, IDLE_EVENTS,
    CATEGORIES, TASK_DEFS_CSV,
)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ActionEvent:
    event: str
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


@dataclass
class Interval:
    obs_frame: int        # start of Reachout (scene before contact)
    mid_frame: int        # start of Grasp (hand on object)
    end_frame: int        # end of manipulation cycle
    events: List[ActionEvent] = field(default_factory=list)


@dataclass
class TrajectoryPoint:
    frame_idx: int
    position_3d: List[float]       # in obs_frame camera coords
    position_pixel: Optional[List[int]] = None


# ============================================================
# Action segmentation -> intervals
# ============================================================

def parse_action_events(action_json_path: Path) -> List[ActionEvent]:
    """Parse action/color.json into a list of ActionEvent."""
    with open(action_json_path) as f:
        data = json.load(f)

    events = []
    for ev in data.get("events", []):
        start_t = ev["startTime"]
        end_t = ev["endTime"]
        events.append(ActionEvent(
            event=ev["event"],
            start_time=start_t,
            end_time=end_t,
            start_frame=int(start_t * FPS),
            end_frame=min(int(end_t * FPS), 299),
        ))
    return events


def detect_intervals(events: List[ActionEvent]) -> List[Interval]:
    """Group action events into manipulation intervals.

    An interval starts at "Reachout"/"Reach" and ends when the next
    idle event ("rest"/"stop") begins or the sequence ends.
    """
    intervals: List[Interval] = []
    current_events: List[ActionEvent] = []
    obs_frame: Optional[int] = None
    mid_frame: Optional[int] = None
    in_manipulation = False

    for ev in events:
        ev_lower = ev.event.lower()

        if not in_manipulation and ev_lower in MANIPULATION_START:
            in_manipulation = True
            obs_frame = ev.start_frame
            mid_frame = None
            current_events = [ev]

        elif in_manipulation:
            current_events.append(ev)

            # Record mid_frame at first Grasp
            if ev_lower == "grasp" and mid_frame is None:
                mid_frame = ev.start_frame

            # End of manipulation cycle
            if ev_lower in IDLE_EVENTS:
                end_frame = current_events[-2].end_frame if len(current_events) >= 2 else ev.start_frame
                if mid_frame is None:
                    mid_frame = (obs_frame + end_frame) // 2

                intervals.append(Interval(
                    obs_frame=obs_frame,
                    mid_frame=mid_frame,
                    end_frame=end_frame,
                    events=current_events[:-1],  # exclude the idle event
                ))
                in_manipulation = False
                current_events = []

    # Handle case where sequence ends during manipulation (no trailing rest/stop)
    if in_manipulation and current_events:
        end_frame = current_events[-1].end_frame
        if mid_frame is None:
            mid_frame = (obs_frame + end_frame) // 2
        intervals.append(Interval(
            obs_frame=obs_frame,
            mid_frame=mid_frame,
            end_frame=end_frame,
            events=current_events,
        ))

    return intervals


# ============================================================
# Camera
# ============================================================

def load_intrinsics(intrin_path: Path) -> Dict[str, float]:
    """Load 3x3 intrinsic matrix -> {fx, fy, cx, cy}."""
    K = np.load(str(intrin_path))
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "resolution": [IMAGE_W, IMAGE_H],
    }


def load_camera_extrinsics(output_log_path: Path) -> List[np.ndarray]:
    """Parse 3Dseg/output.log -> list of 4x4 camera-to-world matrices.

    Format: each entry is a header line (frame_id id1 id2) followed by
    4 lines of 4 floats (4x4 matrix).
    """
    matrices = []
    with open(output_log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # Header line: "frame_id id1 id2"
        header = lines[i].strip().split()
        if len(header) < 3:
            i += 1
            continue

        # Read 4x4 matrix (next 4 lines)
        mat = np.zeros((4, 4), dtype=np.float64)
        for row in range(4):
            vals = lines[i + 1 + row].strip().split()
            mat[row] = [float(v) for v in vals]
        matrices.append(mat)
        i += 5  # header + 4 matrix rows

    return matrices


# ============================================================
# Object pose
# ============================================================

def load_objpose(objpose_dir: Path, frame_idx: int) -> Optional[dict]:
    """Load object pose at a specific frame.

    File naming is NOT zero-padded: 0.json, 1.json, ..., 299.json
    """
    json_path = objpose_dir / f"{frame_idx}.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    if not data.get("isEffective") or not data.get("dataList"):
        return None

    return data


# ============================================================
# Trajectory extraction
# ============================================================

def extract_trajectory(
    objpose_dir: Path,
    extrinsics: List[np.ndarray],
    intrinsics: Dict[str, float],
    obs_frame: int,
    end_frame: int,
    mask_dir: Optional[Path] = None,
) -> Tuple[List[TrajectoryPoint], float]:
    """Extract object trajectory, transform to obs_frame camera coords.

    Uses 2Dseg mask centroid (label 1) as the 2D anchor and objpose depth
    for 3D back-projection. This tracks the *visible moving region* rather
    than the CAD model center, which is critical for:
    - Articulated objects (StorageFurniture, TrashCan, Safe) where objpose
      index 0 is the static base, not the moving part.
    - All objects where the geometric center differs from the contact area.

    Falls back to objpose center when no mask is available at a frame.

    Returns:
        (trajectory_points, max_world_displacement)
    """
    # T_cam_to_world at obs_frame
    if obs_frame < len(extrinsics):
        T_c2w_obs = extrinsics[obs_frame]
    else:
        T_c2w_obs = np.eye(4)

    T_w2c_obs = np.linalg.inv(T_c2w_obs)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    points: List[TrajectoryPoint] = []
    world_positions: List[np.ndarray] = []

    for frame_idx in range(obs_frame, min(end_frame + 1, 300)):
        # Try mask centroid first, fall back to objpose center
        p_cam_i = _get_mask_centroid_3d(
            mask_dir, frame_idx, objpose_dir, intrinsics,
        )
        if p_cam_i is None:
            continue

        # Transform: cam_i -> world -> cam_obs
        if frame_idx < len(extrinsics):
            T_c2w_i = extrinsics[frame_idx]
        else:
            T_c2w_i = np.eye(4)

        p_world = (T_c2w_i @ np.append(p_cam_i, 1.0))[:3]
        p_cam_obs = (T_w2c_obs @ np.append(p_world, 1.0))[:3]

        # Project to pixel
        pixel = None
        if p_cam_obs[2] > 0:
            u = int(round(fx * p_cam_obs[0] / p_cam_obs[2] + cx))
            v = int(round(fy * p_cam_obs[1] / p_cam_obs[2] + cy))
            if 0 <= u < IMAGE_W and 0 <= v < IMAGE_H:
                pixel = [u, v]

        points.append(TrajectoryPoint(
            frame_idx=frame_idx,
            position_3d=p_cam_obs.tolist(),
            position_pixel=pixel,
        ))
        world_positions.append(p_world)

    # Compute max world displacement
    max_disp = 0.0
    if len(world_positions) >= 2:
        positions = np.array(world_positions)
        displacements = np.linalg.norm(positions - positions[0], axis=1)
        max_disp = float(np.max(displacements))

    return points, max_disp


def _get_mask_centroid_3d(
    mask_dir: Optional[Path],
    frame_idx: int,
    objpose_dir: Path,
    intrinsics: Dict[str, float],
) -> Optional[np.ndarray]:
    """Get 3D position by back-projecting the 2Dseg mask centroid using objpose depth.

    Strategy:
    1. Read the 2Dseg mask at frame_idx, compute label-1 centroid (u, v).
    2. Use the z of the objpose dataList entry whose projected center is
       closest to the mask centroid.
    3. Back-project (u, v, z) -> 3D camera coords.

    Falls back to the first objpose center when the mask is missing.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    mask_centroid = None
    if mask_dir is not None:
        mask_path = mask_dir / f"{frame_idx:05d}.png"
        if mask_path.exists():
            raw = np.array(PILImage.open(mask_path))
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            ys, xs = np.where(raw == 1)
            if len(ys) > 0:
                mask_centroid = (float(np.mean(xs)), float(np.mean(ys)))

    # Load objpose for depth reference
    pose = load_objpose(objpose_dir, frame_idx)
    if pose is None:
        return None

    data_list = pose.get("dataList", [])
    if not data_list:
        return None

    if mask_centroid is not None:
        mu, mv = mask_centroid
        # Pick the dataList entry whose projected center is closest to mask centroid
        best_z = None
        best_dist = float("inf")
        for obj in data_list:
            c = obj.get("center")
            if c is None:
                continue
            oz = c["z"]
            if oz <= 0:
                continue
            ou = fx * c["x"] / oz + cx
            ov = fy * c["y"] / oz + cy
            dist = (ou - mu) ** 2 + (ov - mv) ** 2
            if dist < best_dist:
                best_dist = dist
                best_z = oz

        if best_z is None or best_z <= 0:
            return None

        # Back-project mask centroid with the selected depth
        x = (mu - cx) * best_z / fx
        y = (mv - cy) * best_z / fy
        return np.array([x, y, best_z], dtype=np.float64)

    # Fallback: use objpose center of the first entry
    center = data_list[0].get("center")
    if center is None:
        return None
    return np.array([center["x"], center["y"], center["z"]], dtype=np.float64)


# ============================================================
# Mask extraction
# ============================================================

def extract_mask(mask_dir: Path, frame_idx: int, target_label: int = 1) -> Optional[np.ndarray]:
    """Extract binary mask from 2Dseg at given frame.

    Returns uint8 array (0 or 255), or None if missing/empty.

    NOTE: Must use PIL to read -- cv2.imread converts palette PNG colors,
    destroying label values (e.g., label 1 -> pixel value 38).
    """
    mask_path = mask_dir / f"{frame_idx:05d}.png"
    if not mask_path.exists():
        return None

    raw = np.array(PILImage.open(mask_path))
    if raw.ndim == 3:
        raw = raw[:, :, 0]  # shouldn't happen for these masks, but safety

    binary = (raw == target_label).astype(np.uint8) * 255

    if binary.sum() == 0:
        return None

    return binary


# ============================================================
# Frame decoding
# ============================================================

def decode_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """Decode a single RGB frame from video. Returns RGB uint8 HxWx3."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def decode_depth_frame(depth_cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """Decode a single depth frame. Returns float32 in mm."""
    depth_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = depth_cap.read()
    if not ret:
        return None
    # NOTE: VideoCapture decodes to 8-bit frames, so these values are not mm.
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return frame.astype(np.float32)


# ============================================================
# Visualization
# ============================================================

def visualize_trajectory(
    obs_frame: np.ndarray,
    trajectory: List[TrajectoryPoint],
    output_path: Path,
):
    """Draw trajectory on obs_frame and save."""
    img = cv2.cvtColor(obs_frame.copy(), cv2.COLOR_RGB2BGR)

    pixels = [p.position_pixel for p in trajectory if p.position_pixel is not None]
    if not pixels:
        cv2.imwrite(str(output_path), img)
        return

    color = (0, 255, 0)
    for i in range(len(pixels) - 1):
        cv2.line(img, tuple(pixels[i]), tuple(pixels[i + 1]), color, 2)
    for px in pixels:
        cv2.circle(img, tuple(px), 3, color, -1)
    if pixels:
        cv2.circle(img, tuple(pixels[0]), 5, (0, 0, 255), 2)   # start = red
        cv2.circle(img, tuple(pixels[-1]), 5, (255, 0, 0), 2)  # end = blue

    cv2.imwrite(str(output_path), img)


# ============================================================
# Task definitions
# ============================================================

_TASK_DEFS: Optional[Dict[str, Dict[str, str]]] = None


def load_task_definitions() -> Dict[str, Dict[str, str]]:
    """Load task_definitions.csv -> {category_id: {task_id: description}}."""
    global _TASK_DEFS
    if _TASK_DEFS is not None:
        return _TASK_DEFS

    _TASK_DEFS = {}
    with open(TASK_DEFS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat_id = row["Category ID"]
            _TASK_DEFS[cat_id] = {}
            for i in range(1, 7):
                key = f"T{i}"
                val = row.get(key, "").strip()
                if val and val != "N/A":
                    _TASK_DEFS[cat_id][key] = val
    return _TASK_DEFS


def get_task_description(category_id: str, task_id: str) -> str:
    """Get task description, e.g., ('C13', 'T1') -> 'Pick and place'."""
    defs = load_task_definitions()
    return defs.get(category_id, {}).get(task_id, "Manipulate object")


def make_language(category_id: str, task_id: str) -> str:
    """Generate imperative task description.

    e.g., ('C13', 'T1') -> 'Pick up and place the knife.'
    """
    task = get_task_description(category_id, task_id)
    category_name = CATEGORIES.get(category_id, "object").lower()

    # If task already mentions the object, use as-is
    if category_name in task.lower():
        return f"{task}."

    # Replace "it" with the object name
    if " it " in task.lower():
        return task.replace(" it ", f" the {category_name} ").replace(" It ", f" The {category_name} ") + "."

    # Append object name
    return f"{task} the {category_name}."
