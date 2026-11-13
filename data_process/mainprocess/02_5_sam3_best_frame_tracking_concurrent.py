#!/usr/bin/env python3
"""
02_5_sam3_best_frame_tracking_concurrent.py - SAM3 best-frame scan + video tracking.

For each cam view of an interval with a SAM3 text prompt:
  1. Decode the clip [obs_frame, end_frame] to temporary JPEGs.
  2. Scan every scan_stride-th frame with the SAM3 image model and pick the
     frame with the highest text-detection score (fallback: obs_frame).
  3. Text-prompt the SAM3 video predictor on that frame, keep one object
     (highest score; ties within 0.05 broken by distance to the interval
     start point), and propagate it in both directions over the clip.
  4. Save masks to sam3_text_prompt_tracking_masks/frame_NNNNN.png, where N
     is the index in the decoded clip (the offset from obs_frame; stage 04
     relies on this).

Each Ray actor (one per GPU by default) holds `concurrent_videos` predictor
slots and tracks that many videos at once in threads of one CUDA context.

Reads:
    query_gen manifest (sam3_best_frame_track.qg_manifest_path, default utils.QUERY_GEN_MANIFEST_PATH)
    {output_root}/{dataset}/{ep_id}/episode_summary.json, interval_NNN/metadata.json,
    interval_NNN/cam_N/trajectory.json, and the source video via the dataset preprocessor

Writes:
    {output_root}/sam3_video_track_manifest.json
    cam_N/sam3_text_prompt_tracking_masks/, cam_N/sam3_mask_meta.json

Run:
    python mainprocess.py stage=sam3_best_frame_track_concurrent dataset_name=droid
    python mainprocess.py stage=sam3_best_frame_track_concurrent limit=5 sam3_best_frame_track.concurrent_videos=4
"""

import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
try:
    import ray
except ImportError:
    ray = None
import torch
from tqdm import tqdm

import sam3 as _sam3_pkg
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_predictor

from .utils import (
    STAGE_SAM3_VIDEO_TRACK,
    done_keys,
    extract_prompts,
    get_dataset_preprocessor,
    get_dataset_roots,
    load_manifest,
    load_manifest_at,
    manifest_path,
    save_manifest,
    QUERY_GEN_MANIFEST_PATH,
)


@dataclass(frozen=True)
class WorkItem:
    key: str
    dataset_name: str
    episode_id: str
    interval_name: str
    cam_name: str
    cam_dir: Path
    episode_path: str
    cam_id: str
    obs_frame: int
    start_frame: int
    end_frame: int
    text_prompt: str
    start_point: Optional[Tuple[int, int]]


# =============================================================================
# SAM3 video predictor
# =============================================================================


def setup_sam3_video_predictor(confidence_threshold: float = 0.2):
    """Initialize the SAM3 video predictor on the current CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video tracking requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
    if hasattr(predictor, "confidence_threshold"):
        predictor.confidence_threshold = confidence_threshold
    return predictor


def setup_sam3_image_processor() -> Sam3Processor:
    """Initialize SAM3 image model for fast per-frame text detection."""
    sam3_root = os.path.join(os.path.dirname(_sam3_pkg.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    return Sam3Processor(model, confidence_threshold=0.0)


def _patch_triton_autotuner_thread_safety() -> None:
    """Monkey-patch Triton's Autotuner to be thread-safe.

    Triton's ``@autotune`` benchmarking loop does
    ``{**self.nargs, **current}`` (autotuner.py:143) where ``self.nargs``
    can be None when two threads autotune the same kernel simultaneously.
    We wrap ``Autotuner.run`` with a global lock so only one thread can
    autotune at a time.  Once a kernel's configs are cached the lock is
    never contended.
    """
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return

    if getattr(Autotuner, "_thread_safety_patched", False):
        return

    _lock = threading.Lock()
    _orig_run = Autotuner.run

    def _safe_run(self, *args, **kwargs):
        with _lock:
            return _orig_run(self, *args, **kwargs)

    Autotuner.run = _safe_run
    Autotuner._thread_safety_patched = True
    print("  Patched Triton autotuner for thread safety.")


def scan_best_frame(
    image_processor: Sam3Processor,
    frame_dir: Path,
    text_prompt: str,
    num_frames: int,
    stride: int = 10,
) -> Tuple[Optional[int], float]:
    """
    Scan video frames with the image model to find the frame where the
    text-prompted object has the highest detection confidence.

    Returns (best_frame_idx, best_score) or (None, 0.0) if nothing found.
    """
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    best_score = 0.0
    best_frame: Optional[int] = None

    for fi in range(0, min(num_frames, len(frame_paths)), stride):
        img = Image.open(str(frame_paths[fi])).convert("RGB")
        try:
            state = image_processor.set_image(img)
            if state is None:
                continue
            state = image_processor.set_text_prompt(state=state, prompt=text_prompt)
            if state is None:
                continue
        except (TypeError, RuntimeError):
            continue

        scores = state.get("scores", None)
        if scores is not None and len(scores) > 0:
            max_score = float(scores.float().cpu().max())
            if max_score > best_score:
                best_score = max_score
                best_frame = fi

        if state is not None:
            image_processor.reset_all_prompts(state)

    return best_frame, best_score


# =============================================================================
# Discovery
# =============================================================================


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_start_point(traj: dict, start_frame: int) -> Optional[Tuple[int, int]]:
    """Get a seed pixel aligned to the interval start frame."""
    for item in traj.get("trajectory_2d", []):
        if item.get("frame_idx") == start_frame and item.get("position_pixel") is not None:
            x, y = item["position_pixel"]
            return int(round(x)), int(round(y))

    motion_start = (traj.get("motion_2d") or {}).get("start_pixel")
    if motion_start is not None:
        x, y = motion_start
        return int(round(x)), int(round(y))

    for item in traj.get("trajectory_2d", []):
        if item.get("position_pixel") is not None:
            x, y = item["position_pixel"]
            return int(round(x)), int(round(y))

    return None


def discover_work(
    dataset_roots: List[Path],
    prompts: Dict[str, str],
    existing_keys: set,
    shuffle: bool = False,
    seed: Optional[int] = 42,
    skip_existing: bool = False,
    limit: Optional[int] = None,
) -> List[WorkItem]:
    """
    Discover camera tracks to process.

    Prompts-driven: we iterate only the intervals referenced by the qg
    manifest (``prompts``) instead of walking every episode directory on
    disk. For fully-done intervals we skip both ``metadata.json`` and
    ``trajectory.json`` reads (fast-path via ``existing_keys``).

    Returns:
        list of WorkItem, keyed by "episode_id/interval_name/cam_name".
    """
    items: List[WorkItem] = []

    # Group valid prompts by episode so we touch each episode_summary.json once.
    prompts_by_ep: Dict[str, List[Tuple[str, str]]] = {}
    for iv_key, prompt in prompts.items():
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        parts = iv_key.split("/", 1)
        if len(parts) != 2:
            continue
        ep_id, iv_name = parts
        prompts_by_ep.setdefault(ep_id, []).append((iv_name, prompt.strip()))

    if not prompts_by_ep:
        return items

    # Fast-path: drop episodes whose every (iv, cam_0) key is already done,
    # so their files are never touched. Multi-camera episodes are not skipped
    # here and fall through to the per-cam check below.
    if skip_existing and existing_keys:
        fully_done_eps: List[str] = []
        for ep_id, iv_list in prompts_by_ep.items():
            if all(
                f"{ep_id}/{iv_name}/cam_0" in existing_keys
                for iv_name, _ in iv_list
            ):
                fully_done_eps.append(ep_id)
        for ep_id in fully_done_eps:
            del prompts_by_ep[ep_id]
        if fully_done_eps:
            print(
                f"[discover_work] Fast-path: skipped {len(fully_done_eps)} "
                f"fully-done episodes ({len(prompts_by_ep)} remain to scan)"
            )
        if not prompts_by_ep:
            return items

    # Locate each referenced episode in the dataset roots. We only stat the
    # episodes named in prompts, not every directory on disk.
    ep_lookup: Dict[str, Tuple[str, Path, Dict[str, Any]]] = {}
    for ep_id in prompts_by_ep.keys():
        for root in dataset_roots:
            ep_dir = root / ep_id
            summary_path = ep_dir / "episode_summary.json"
            if not summary_path.exists():
                continue
            episode_summary = _load_json(summary_path)
            if episode_summary is None:
                continue
            episode_path = episode_summary.get("episode_path")
            if not episode_path:
                continue
            ep_lookup[ep_id] = (root.name, ep_dir, episode_summary)
            break

    ep_iter = list(prompts_by_ep.items())
    if shuffle:
        import random
        rng = random.Random(seed)
        rng.shuffle(ep_iter)

    for ep_id, iv_list in tqdm(ep_iter, desc="discover_work"):
        hit = ep_lookup.get(ep_id)
        if hit is None:
            continue
        dataset_name, ep_dir, episode_summary = hit
        episode_path = episode_summary["episode_path"]

        for iv_name, text_prompt in iv_list:
            iv_dir = ep_dir / iv_name
            if not iv_dir.is_dir():
                continue

            # Fast-path: list cam dirs first. If skip_existing and all cams
            # are already done, avoid loading metadata + trajectory entirely.
            cam_dirs_all = sorted(
                d for d in iv_dir.iterdir()
                if d.is_dir() and d.name.startswith("cam_")
            )
            if not cam_dirs_all:
                continue

            if skip_existing:
                pending_cam_dirs = [
                    d for d in cam_dirs_all
                    if f"{ep_id}/{iv_name}/{d.name}" not in existing_keys
                ]
                if not pending_cam_dirs:
                    continue
            else:
                pending_cam_dirs = cam_dirs_all

            metadata_path = iv_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = _load_json(metadata_path)
            if metadata is None:
                continue

            obs_frame = metadata.get("obs_frame")
            start_frame = metadata.get("start_frame")
            end_frame = metadata.get("end_frame")
            if obs_frame is None:
                obs_frame = 0
            if (
                start_frame is None
                or end_frame is None
                or end_frame < start_frame
                or start_frame < obs_frame
            ):
                continue

            for cam_dir in pending_cam_dirs:
                cam_name = cam_dir.name
                key = f"{ep_id}/{iv_name}/{cam_name}"
                if skip_existing and key in existing_keys:
                    continue

                traj_path = cam_dir / "trajectory.json"
                if not traj_path.exists():
                    continue
                traj = _load_json(traj_path)
                if traj is None:
                    continue

                # May be None (no projected end-effector): then selection is score-only.
                start_point = _extract_start_point(traj, start_frame)

                cam_id = metadata.get(cam_name, {}).get("cam_id")
                if not cam_id:
                    cam_id = traj.get("camera_info", {}).get("cam_id")
                if not cam_id:
                    continue

                items.append(WorkItem(
                    key=key,
                    dataset_name=dataset_name,
                    episode_id=ep_id,
                    interval_name=iv_name,
                    cam_name=cam_name,
                    cam_dir=cam_dir,
                    episode_path=episode_path,
                    cam_id=cam_id,
                    obs_frame=int(obs_frame),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    text_prompt=text_prompt.strip(),
                    start_point=start_point,
                ))

                if limit is not None and len(items) >= limit:
                    return items

    return items


# =============================================================================
# Interval frame extraction
# =============================================================================


def _prepare_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


def _save_jpeg_frame(rgb: np.ndarray, output_path: Path, jpeg_quality: int = 95) -> None:
    Image.fromarray(rgb).save(output_path, format="JPEG", quality=jpeg_quality)


def build_interval_frame_dir(
    preprocessor,
    item: WorkItem,
    temp_dir: Path,
    jpeg_quality: int = 95,
) -> Dict[int, int]:
    """
    Reconstruct the interval clip into a temporary frame directory.

    Returns:
        Mapping from interval-local frame index to global episode frame index.
    """
    _prepare_temp_dir(temp_dir)
    local_to_global: Dict[int, int] = {}

    # Temporarily disable depth extraction — this stage only needs RGB.
    _saved_depth = getattr(preprocessor, '_extract_depth', None)
    if _saved_depth is not None:
        preprocessor._extract_depth = False

    episode_data = preprocessor.load_episode(item.episode_path)

    skipped = 0
    try:
        saved_idx = 0
        for frame_idx in range(item.obs_frame, item.end_frame + 1):
            try:
                rgb = preprocessor.extract_frame(episode_data, item.cam_id, frame_idx)
                if rgb is None:
                    raise ValueError("extract_frame returned None")
                frame_path = temp_dir / f"{saved_idx:05d}.jpg"
                _save_jpeg_frame(rgb, frame_path, jpeg_quality=jpeg_quality)
            except Exception as e:
                skipped += 1
                print(f"  [WARN] Skipping frame {frame_idx}: {e}")
                continue
            local_to_global[saved_idx] = frame_idx
            saved_idx += 1
    finally:
        preprocessor.close_episode(episode_data)
        if _saved_depth is not None:
            preprocessor._extract_depth = _saved_depth

    if skipped:
        print(f"  [WARN] Skipped {skipped} unreadable frames, {saved_idx} frames saved")

    return local_to_global


def _prepare_work_item(
    item: WorkItem,
    output_root: str,
    temp_frame_root: str,
    jpeg_quality: int = 95,
) -> dict:
    """Materialize a trackable interval clip on disk."""
    output_root_path = Path(output_root)
    temp_dir = Path(temp_frame_root) / item.dataset_name / item.episode_id / item.interval_name / item.cam_name

    try:
        preprocessor = get_dataset_preprocessor(item.dataset_name, output_root_path, extract_depth=False)
        local_to_global = build_interval_frame_dir(
            preprocessor,
            item,
            temp_dir,
            jpeg_quality=jpeg_quality,
        )
        return {
            "item": item,
            "temp_dir": str(temp_dir),
            "local_to_global": local_to_global,
            "prompt_frame_idx": 0,  # obs_frame in local coords
        }
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return {
            "item": item,
            "error": f"{type(e).__name__}: {e}",
        }


# =============================================================================
# Tracking helpers
# =============================================================================

_OOM_MAX_RETRIES = 3
_OOM_BACKOFF_SECS = 2


def _is_oom_error(reason: str) -> bool:
    """Check if an error reason string indicates a CUDA out-of-memory error."""
    reason_lower = reason.lower()
    return "cuda out of memory" in reason_lower or "outofmemoryerror" in reason_lower


def _as_bool_mask(mask: Any) -> np.ndarray:
    mask_np = np.asarray(mask)
    mask_np = np.squeeze(mask_np)
    return mask_np > 0


def _extract_object_mask(outputs: dict, obj_id: int) -> Optional[np.ndarray]:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks", [])
    for idx, candidate_obj_id in enumerate(obj_ids):
        if int(candidate_obj_id) == int(obj_id):
            return _as_bool_mask(masks[idx])
    return None


def _mask_centroid_distance(mask: np.ndarray, point: Tuple[int, int]) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("inf")
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    return float(np.hypot(centroid_x - point[0], centroid_y - point[1]))


def select_object_from_prompt_outputs(
    outputs: dict,
    start_point: Optional[Tuple[int, int]],
    score_tie_threshold: float = 0.05,
) -> Optional[dict]:
    """Pick the best text-detected object, prioritizing score.

    Selection logic:
      1. Pick the highest-scoring candidate.
      2. If *start_point* exists and is in-frame, collect all candidates
         whose score is within *score_tie_threshold* of the top score.
         Among those, pick the one closest to start_point.
      3. Otherwise return the top scorer.
    """
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks", [])
    probs = np.asarray(outputs.get("out_probs", np.zeros(len(obj_ids), dtype=np.float32)))

    if len(obj_ids) == 0 or len(masks) == 0:
        return None

    # Build candidates
    candidates: List[dict] = []
    for idx, obj_id in enumerate(obj_ids):
        mask = _as_bool_mask(masks[idx])
        if mask.ndim != 2:
            continue
        score = float(probs[idx]) if idx < len(probs) else 0.0
        if start_point is not None:
            x, y = int(start_point[0]), int(start_point[1])
            h, w = mask.shape
            contains_point = 0 <= x < w and 0 <= y < h and bool(mask[y, x])
            distance_px = 0.0 if contains_point else _mask_centroid_distance(mask, (x, y))
        else:
            contains_point = False
            distance_px = 0.0
        candidates.append({
            "obj_id": int(obj_id),
            "score": score,
            "contains_point": contains_point,
            "distance_px": distance_px,
            "mask": mask,
        })

    if not candidates:
        return None

    # Primary selection: highest score
    best = max(candidates, key=lambda c: c["score"])

    # Tiebreak with distance when start_point is available and in-frame
    if start_point is not None:
        x, y = int(start_point[0]), int(start_point[1])
        h, w = candidates[0]["mask"].shape[:2]
        point_in_frame = 0 <= x < w and 0 <= y < h
        if point_in_frame:
            tied = [c for c in candidates if c["score"] >= best["score"] - score_tie_threshold]
            selected = min(tied, key=lambda c: c["distance_px"])
        else:
            selected = best
    else:
        selected = best

    return selected


def _track_one_direction(
    predictor,
    frame_dir: Path,
    text_prompt: str,
    start_point: Optional[Tuple[int, int]],
    prompt_frame_idx: int,
    direction: str,
) -> dict:
    """
    Run SAM3 text-prompted tracking on the seed frame, select the matching
    object there, then propagate in one direction ("forward" or "backward")
    in a session of its own.
    """
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=str(frame_dir),
        )
    )
    session_id = response["session_id"]

    try:
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame_idx,
                text=text_prompt,
            )
        )
        if response is None:
            return {
                "status": "done", "has_track": False,
                "reason": "predictor_returned_none_on_add_prompt",
            }
        prompt_outputs = response.get("outputs")
        if prompt_outputs is None:
            return {
                "status": "done", "has_track": False,
                "reason": "prompt_outputs_none",
            }
        selected = select_object_from_prompt_outputs(prompt_outputs, start_point)

        if selected is None:
            return {
                "status": "done",
                "has_track": False,
                "reason": "no_text_prompt_objects",
            }

        selected_obj_id = int(selected["obj_id"])
        selected_mask = selected["mask"]

        for obj_id in np.asarray(prompt_outputs.get("out_obj_ids", [])).tolist():
            obj_id = int(obj_id)
            if obj_id != selected_obj_id:
                predictor.handle_request(
                    request=dict(
                        type="remove_object",
                        session_id=session_id,
                        obj_id=obj_id,
                        is_user_action=False,
                    )
                )

        masks_by_frame: Dict[int, np.ndarray] = {prompt_frame_idx: selected_mask}
        for frame_response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=direction,
                start_frame_index=prompt_frame_idx,
            )
        ):
            frame_idx = int(frame_response["frame_index"])
            outputs = frame_response.get("outputs")
            if outputs is None:
                continue
            mask = _extract_object_mask(outputs, selected_obj_id)
            if mask is not None:
                masks_by_frame[frame_idx] = mask

        return {
            "status": "done",
            "has_track": len(masks_by_frame) > 0,
            "selected_obj_id": selected_obj_id,
            "contains_start_point": bool(selected["contains_point"]),
            "selection_distance_px": float(selected["distance_px"]),
            "masks_by_frame": masks_by_frame,
        }
    finally:
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )


def track_interval_video(
    predictor,
    frame_dir: Path,
    text_prompt: str,
    start_point: Optional[Tuple[int, int]],
    prompt_frame_idx: int,
) -> dict:
    """
    Track the text-prompted object over the whole interval clip.

    The forward and the backward pass each run in a session of their own. With
    propagation_direction="both" in one session, SAM3 starts the backward pass
    from the tracker state left at the end of the forward pass, where the
    selected object is often gone. No mask then came back for the frames before
    the prompt frame, so the observation frame (frame 0) had no mask.
    """
    result = _track_one_direction(
        predictor, frame_dir, text_prompt, start_point, prompt_frame_idx, "forward",
    )
    if not result.get("has_track") or prompt_frame_idx == 0:
        return result

    backward = _track_one_direction(
        predictor, frame_dir, text_prompt, start_point, prompt_frame_idx, "backward",
    )
    forward_mask = _as_bool_mask(result["masks_by_frame"][prompt_frame_idx])
    backward_masks = backward.get("masks_by_frame") or {}
    backward_mask = backward_masks.get(prompt_frame_idx)
    if backward_mask is None:
        return result
    backward_mask = _as_bool_mask(backward_mask)
    union = np.logical_or(forward_mask, backward_mask).sum()
    same_object = union > 0 and np.logical_and(forward_mask, backward_mask).sum() / union > 0.5
    if same_object:
        for frame_idx, mask in backward_masks.items():
            result["masks_by_frame"].setdefault(frame_idx, mask)
    return result


# =============================================================================
# Artifact saving
# =============================================================================


def save_masks_to_folder(masks_by_frame: Dict[int, np.ndarray], output_dir: Path) -> int:
    """Persist binary masks as PNG files in a dedicated folder."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_idx, mask in sorted(masks_by_frame.items()):
        mask_uint8 = (_as_bool_mask(mask).astype(np.uint8) * 255)
        Image.fromarray(mask_uint8).save(output_dir / f"frame_{frame_idx:05d}.png")
        saved += 1
    return saved


def save_mask_meta(output_path: Path, stride: int = 1) -> None:
    """Save minimal mask metadata for downstream consumers."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"stride": int(stride)}, f, indent=4)


def _entry_base(item: WorkItem) -> dict:
    return {
        "prompt": item.text_prompt,
        "obs_frame": item.obs_frame,
        "start_frame": item.start_frame,
        "end_frame": item.end_frame,
        "num_interval_frames": item.end_frame - item.start_frame + 1,
        "prompt_frame_index": 0,
        "used_start_point_refinement": False,
    }


def _error_entry(item: WorkItem, reason: str) -> dict:
    entry = _entry_base(item)
    entry.update({
        "status": "error",
        "has_track": False,
        "reason": reason,
        "selected_obj_id": None,
        "contains_start_point": None,
        "selection_distance_px": None,
        "num_saved_masks": 0,
        "mask_dir": None,
        "mask_meta_path": None,
        "saved_frame_indices": [],
    })
    return entry


def _track_prepared_item(
    predictor,
    image_processor: Sam3Processor,
    prepared: dict,
    output_root: Path,
    scan_stride: int = 10,
) -> dict:
    item = prepared["item"]
    temp_dir = Path(prepared["temp_dir"])
    local_to_global = prepared["local_to_global"]
    fallback_frame_idx = int(prepared["prompt_frame_idx"])
    num_frames = len(list(temp_dir.glob("*.jpg")))
    mask_dir = item.cam_dir / "sam3_text_prompt_tracking_masks"
    mask_meta_path = item.cam_dir / "sam3_mask_meta.json"

    try:
        # ── scan for best seed frame ──
        best_frame, best_score = scan_best_frame(
            image_processor, temp_dir, item.text_prompt, num_frames,
            stride=scan_stride,
        )
        if best_frame is not None:
            prompt_frame_idx = best_frame
        else:
            prompt_frame_idx = fallback_frame_idx

        # ── track from best frame ──
        tracking = track_interval_video(
            predictor,
            temp_dir,
            item.text_prompt,
            item.start_point,
            prompt_frame_idx=prompt_frame_idx,
        )

        num_saved_masks = 0
        mask_dir_rel = None
        # Bidirectional propagation covers the whole clip [0, num_frames).
        # Masks are saved by local index N; stage 04 maps N to video frame
        # obs_frame + N and clips to [obs_frame, end_frame].
        saved_masks_by_frame = tracking.get("masks_by_frame", {})

        if tracking.get("has_track"):
            num_saved_masks = save_masks_to_folder(saved_masks_by_frame, mask_dir)
            save_mask_meta(mask_meta_path, stride=1)
            mask_dir_rel = os.path.relpath(mask_dir, output_root)
        elif mask_dir.exists():
            shutil.rmtree(mask_dir)
            if mask_meta_path.exists():
                mask_meta_path.unlink()
        elif mask_meta_path.exists():
            mask_meta_path.unlink()

        entry = _entry_base(item)
        entry.update({
            "status": tracking["status"],
            "has_track": bool(tracking.get("has_track")),
            "reason": tracking.get("reason"),
            "selected_obj_id": tracking.get("selected_obj_id"),
            "contains_start_point": tracking.get("contains_start_point"),
            "selection_distance_px": tracking.get("selection_distance_px"),
            "num_saved_masks": num_saved_masks,
            "mask_dir": mask_dir_rel,
            "mask_meta_path": (
                os.path.relpath(mask_meta_path, output_root)
                if tracking.get("has_track")
                else None
            ),
            "saved_frame_indices": sorted(
                local_to_global.get(i, i)
                for i in saved_masks_by_frame
            ),
            "scan_best_frame": best_frame,
            "scan_best_score": best_score,
            "scan_stride": scan_stride,
        })
        return entry
    except Exception as e:
        return _error_entry(item, f"{type(e).__name__}: {e}")


class ConcurrentBestFrameTrackWorker:
    """GPU worker with N predictor slots for concurrent video tracking.

    Each slot has its own SAM3 video predictor + image processor.
    Ray calls process_item from multiple threads (max_concurrency),
    and each call acquires a free slot.  All threads share one CUDA
    context, so this works under Exclusive_Process GPU mode.
    """

    def __init__(
        self,
        output_root: str,
        confidence_threshold: float,
        n_concurrent: int = 4,
        score_threshold_detection: float = 0.01,
        scan_stride: int = 10,
    ) -> None:
        torch.cuda.set_device(0)
        self.output_root = Path(output_root)
        self.n_concurrent = n_concurrent
        self.scan_stride = scan_stride

        # Create N independent predictor+image_processor pairs
        self._slot_queue: queue.Queue[int] = queue.Queue()
        self.predictors: List[Any] = []
        self.image_processors: List[Any] = []
        for i in range(n_concurrent):
            print(f"  Initializing predictor slot {i+1}/{n_concurrent}...")
            pred = setup_sam3_video_predictor(
                confidence_threshold=confidence_threshold,
            )
            if hasattr(pred, "model") and hasattr(pred.model, "score_threshold_detection"):
                pred.model.score_threshold_detection = score_threshold_detection
            self.predictors.append(pred)
            self.image_processors.append(setup_sam3_image_processor())
            self._slot_queue.put(i)
            if i == 0:
                _patch_triton_autotuner_thread_safety()
        print(f"  All {n_concurrent} predictor slots ready.")

    def process_item(self, item: "WorkItem", temp_frame_root: str,
                     jpeg_quality: int) -> dict:
        """Full pipeline for one item, using a free predictor slot."""
        slot = self._slot_queue.get()
        prepared = None
        try:
            prepared = _prepare_work_item(
                item, str(self.output_root), temp_frame_root, jpeg_quality,
            )
            if "error" in prepared:
                return _error_entry(item, prepared["error"])
            # autocast is thread-local; must re-enter in each worker thread
            entry = None
            for attempt in range(_OOM_MAX_RETRIES):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Temp frames stay on disk for OOM retries; removed below.
                    entry = _track_prepared_item(
                        self.predictors[slot], self.image_processors[slot],
                        prepared, self.output_root,
                        scan_stride=self.scan_stride,
                    )
                if (entry.get("status") == "error"
                        and _is_oom_error(entry.get("reason", ""))
                        and attempt < _OOM_MAX_RETRIES - 1):
                    print(f"  [OOM] attempt {attempt + 1}/{_OOM_MAX_RETRIES}, "
                          f"clearing cache and retrying in "
                          f"{_OOM_BACKOFF_SECS * (attempt + 1)}s: {item.key}")
                    torch.cuda.empty_cache()
                    time.sleep(_OOM_BACKOFF_SECS * (attempt + 1))
                    continue
                break
            return entry
        finally:
            self._slot_queue.put(slot)
            if prepared and "temp_dir" in prepared:
                temp_dir = Path(prepared["temp_dir"])
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    def shutdown(self) -> None:
        for pred in self.predictors:
            pred.shutdown()
        self.predictors.clear()
        self.image_processors.clear()
        torch.cuda.empty_cache()


# =============================================================================
# Entry point
# =============================================================================


def run_sam3_best_frame_track_concurrent(cfg):
    """Stage 02.5: best-frame scan + video tracking with SAM3."""
    track_cfg = cfg.sam3_best_frame_track
    output_root = Path(cfg.output_root)
    dataset_roots = get_dataset_roots(cfg)
    skip_existing = cfg.skip_existing
    limit = cfg.get("limit")
    flush_every = 10
    num_processes = max(1, int(track_cfg.get("num_processes", 1)))
    concurrent_videos = max(1, int(track_cfg.get("concurrent_videos", 4)))
    jpeg_quality = int(track_cfg.jpeg_quality)
    scan_stride = int(track_cfg.get("scan_stride", 10))
    score_threshold_detection = float(track_cfg.get("score_threshold_detection", 0.01))

    print("=" * 70)
    print("Stage 02.5 - SAM3 Best-Frame Scan + Video Track")
    print(f"  output_root              : {output_root}")
    print(f"  dataset dirs             : {[d.name for d in dataset_roots]}")
    print(f"  scan_stride              : {scan_stride}")
    print(f"  concurrent_videos/GPU    : {concurrent_videos}")
    print(f"  score_threshold_detection: {score_threshold_detection}")
    print(f"  manifest                 : {manifest_path(output_root, STAGE_SAM3_VIDEO_TRACK)}")
    print("=" * 70)

    _qg_override = track_cfg.get("qg_manifest_path", None)
    qg_manifest_path = Path(_qg_override) if _qg_override else QUERY_GEN_MANIFEST_PATH
    print(f"  qg_manifest_path         : {qg_manifest_path}")
    qg_manifest = load_manifest_at(qg_manifest_path)
    if not qg_manifest:
        raise FileNotFoundError(
            f"query_gen manifest not found: {qg_manifest_path}"
        )
    prompts = extract_prompts(qg_manifest)
    print(f"Loaded {len(prompts)} prompts from query_gen manifest")

    # Use the loaded manifest only to derive the skip-set; track NEW entries
    # this run produces in a fresh dict so save_manifest pushes only our diff.
    # Otherwise multiple shards' periodic full saves race and overwrite each
    # other's newer updates with their stale local snapshots.
    _existing_manifest = load_manifest(output_root, STAGE_SAM3_VIDEO_TRACK) if skip_existing else {}
    existing = done_keys(_existing_manifest) if skip_existing else set()
    if existing:
        print(f"Loaded {len(existing)} existing entries from sam3_video_track manifest")
    manifest: Dict[str, Any] = {}

    shuffle = cfg.get("shuffle", False)
    seed = cfg.get("seed", 42)

    items = discover_work(
        dataset_roots,
        prompts,
        existing,
        shuffle,
        seed=seed,
        skip_existing=skip_existing,
        limit=limit,
    )
    print(f"Cam directories to track: {len(items)}")
    if not items:
        print("Nothing to do.")
        return

    total_gpus = torch.cuda.device_count()
    if total_gpus < 1:
        raise RuntimeError("SAM3 video tracking requires at least one CUDA GPU")

    temp_frame_root = (
        Path(track_cfg.temp_frame_root)
        if track_cfg.get("temp_frame_root")
        else output_root / "_sam3_video_track_tmp"
    )

    t0 = time.time()
    tracked_ok = 0
    new_count = 0
    started_ray = False
    gpu_workers: List[Any] = []

    def record_entry(item: WorkItem, entry: dict) -> None:
        nonlocal tracked_ok, new_count
        manifest[item.key] = entry
        new_count += 1
        if entry["has_track"]:
            tracked_ok += 1
        if new_count % flush_every == 0:
            save_manifest(output_root, STAGE_SAM3_VIDEO_TRACK, manifest)

    try:
        if ray is None:
            raise ImportError(
                "ray is required for concurrent tracking. Install ray."
            )
        num_workers = max(1, min(num_processes * total_gpus, len(items)))
        # Need enough CPUs for the concurrent threads inside each actor
        num_cpus = max(num_workers, num_workers * concurrent_videos)
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_gpus=total_gpus,
                     num_cpus=num_cpus)
            started_ray = True
        gpus_per_worker = total_gpus / num_workers
        TrackActor = ray.remote(
            num_cpus=1, num_gpus=gpus_per_worker,
            max_concurrency=concurrent_videos,
        )(ConcurrentBestFrameTrackWorker)
        gpu_workers = [
            TrackActor.remote(
                str(output_root),
                float(track_cfg.confidence_threshold),
                concurrent_videos,
                score_threshold_detection,
                scan_stride,
            )
            for _ in range(num_workers)
        ]
        total_concurrent = num_workers * concurrent_videos
        print(
            f"Spawned {num_workers} SAM3 actor(s) "
            f"({num_processes} per GPU, {gpus_per_worker:.2f} GPU each, "
            f"{concurrent_videos} concurrent videos each = "
            f"{total_concurrent} total, scan_stride={scan_stride}), "
            f"initializing predictors + image model...\n"
        )

        # Seed each actor with concurrent_videos items
        item_idx = 0
        pending: Dict[Any, Tuple[int, WorkItem]] = {}
        for wi, worker in enumerate(gpu_workers):
            for _ in range(concurrent_videos):
                if item_idx >= len(items):
                    break
                ref = worker.process_item.remote(
                    items[item_idx], str(temp_frame_root), jpeg_quality,
                )
                pending[ref] = (wi, items[item_idx])
                item_idx += 1

        errors = 0
        pbar = tqdm(total=len(items), desc="Tracking", unit="vid",
                    dynamic_ncols=True)
        while pending:
            done, _ = ray.wait(list(pending), num_returns=1)
            ref = done[0]
            worker_idx, item = pending.pop(ref)

            try:
                entry = ray.get(ref)
            except Exception as e:
                entry = _error_entry(item, f"{type(e).__name__}: {e}")
            record_entry(item, entry)

            if entry.get("has_track"):
                status_icon = "ok"
            elif entry.get("status") == "error":
                errors += 1
                status_icon = "ERR"
            else:
                status_icon = "no_track"

            pbar.set_postfix_str(
                f"ok={tracked_ok} err={errors} | {status_icon}: {item.key}",
                refresh=False,
            )
            pbar.update(1)

            # Feed the freed slot its next item (same actor)
            if item_idx < len(items):
                ref = gpu_workers[worker_idx].process_item.remote(
                    items[item_idx], str(temp_frame_root), jpeg_quality,
                )
                pending[ref] = (worker_idx, items[item_idx])
                item_idx += 1
        pbar.close()
    finally:
        if gpu_workers:
            try:
                ray.get([worker.shutdown.remote() for worker in gpu_workers])
            except Exception:
                pass
        if started_ray:
            ray.shutdown()

    save_manifest(output_root, STAGE_SAM3_VIDEO_TRACK, manifest)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done: {tracked_ok}/{len(items)} cam views tracked in {elapsed:.1f}s")
    print(f"Manifest -> {manifest_path(output_root, STAGE_SAM3_VIDEO_TRACK)}")
    print(f"{'=' * 70}")
