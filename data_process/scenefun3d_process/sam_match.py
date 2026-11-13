"""Stage 3: SAM3 masks paired with the annotated motions.

Per cam: run SAM3 (image mode) on obs_frame.png with the interval's query-gen prompt and keep
masks scoring >= confidence. Then, for each trajectory in order, among unpaired masks whose
distance to the trajectory start is <= threshold_2d_px in the image AND <= threshold_3d_m in 3D
(mask pixels back-projected with depth), take the highest-scoring one. A cam is valid only when
every trajectory gets a mask; sam_mask.png is then the union of the paired masks.

Writes sam_mask.png per valid cam and {output_root}/scenefun3d_sam_match_manifest.json
({cam key: {status: valid|invalid, reason, ...}}). Run before refine-depth: the 3D gate uses
the rendered depth. Cams already in the manifest are skipped, so reruns resume.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mainprocess.utils import load_manifest_at, save_manifest_at

from .frame_selection import SOURCE, _project_cam

MANIFEST_NAME = "scenefun3d_sam_match_manifest.json"
QUERY_MANIFEST_NAME = "scenefun3d_query_gen_manifest.json"
SAVE_EVERY = 500  # cams between manifest saves


def setup_sam3(confidence_threshold: float):
    import sam3
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def sam_candidates(processor, image_pil, prompt: str, confidence_threshold: float) -> list:
    """[(mask bool HxW, score)] with score >= confidence_threshold."""
    state = processor.set_image(image_pil)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = state.get("masks"), state.get("scores")
    processor.reset_all_prompts(state)
    if masks is None or scores is None or len(masks) == 0:
        return []
    masks_np = masks.squeeze(1).cpu().float().numpy() > 0.5
    scores_np = scores.cpu().float().numpy()
    return [(m.astype(bool), float(s)) for m, s in zip(masks_np, scores_np) if float(s) >= confidence_threshold]


def dist_2d(mask: np.ndarray, uv) -> float:
    """Pixel distance from uv to the mask (0 inside); inf for an empty mask or uv outside the image."""
    if not mask.any():
        return float("inf")
    H, W = mask.shape
    u, v = uv
    if not (0 <= u < W and 0 <= v < H):
        return float("inf")
    u_i, v_i = min(int(round(u)), W - 1), min(int(round(v)), H - 1)
    if mask[v_i, u_i]:
        return 0.0
    return float(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)[v_i, u_i])


def dist_3d(mask: np.ndarray, depth_m: np.ndarray, intr: dict, point_cam: np.ndarray) -> float:
    """Min distance (m) from point_cam to the back-projected valid-depth mask pixels; inf if none."""
    valid = mask & (depth_m > 0)
    if not valid.any():
        return float("inf")
    vs, us = np.where(valid)
    zs = depth_m[vs, us].astype(np.float64)
    xs = (us.astype(np.float64) - float(intr["cx"])) * zs / float(intr["fx"])
    ys = (vs.astype(np.float64) - float(intr["cy"])) * zs / float(intr["fy"])
    pts = np.stack([xs, ys, zs], axis=1)
    return float(np.linalg.norm(pts - np.asarray(point_cam, dtype=np.float64), axis=1).min())


def greedy_match(candidates: list, trajectories: list, depth_m, intr: dict, t2d: float, t3d: float) -> list:
    """Per trajectory (in order): highest-scoring unpaired candidate passing both gates, or None."""
    pairings = [None] * len(trajectories)
    unpaired = list(range(len(candidates)))
    for t_idx, traj in enumerate(trajectories):
        start_3d = np.asarray(traj["trajectory_3d"][0]["position_3d"], dtype=np.float64)
        start_2d = _project_cam(start_3d, intr)
        if start_2d is None:
            continue
        qualifying = []
        for i in unpaired:
            mask, score = candidates[i]
            d2 = dist_2d(mask, start_2d)
            if d2 > t2d:
                continue
            d3 = dist_3d(mask, depth_m, intr, start_3d)
            if d3 > t3d:
                continue
            qualifying.append((i, float(score), float(d2), float(d3)))
        if qualifying:
            i, score, d2, d3 = max(qualifying, key=lambda q: q[1])
            pairings[t_idx] = {"mask_idx": i, "score": score, "dist_2d_px": d2, "dist_3d_m": d3}
            unpaired.remove(i)
    return pairings


def cam_key(cam_dir: Path) -> str:
    """<episode>__<interval idx>__cam_<k> (frame-selection manifest key)."""
    return f"{cam_dir.parent.parent.name}__{cam_dir.parent.name[len('interval_'):]}__{cam_dir.name}"


def process_cam(cam_dir: Path, prompts: dict, processor, conf: float, t2d: float, t3d: float) -> dict:
    """Run SAM3 + pairing for one cam; writes sam_mask.png when every trajectory is paired."""
    mask_out = cam_dir / "sam_mask.png"
    mask_out.unlink(missing_ok=True)  # never keep a mask from an earlier run
    tj = json.loads((cam_dir / "trajectory.json").read_text())
    trajs = tj["trajectories"]
    prompt = prompts.get(f"{cam_dir.parent.parent.name}/{cam_dir.parent.name}")
    if not prompt:
        return {"status": "invalid", "reason": "missing_prompts", "K": len(trajs)}

    image = Image.open(cam_dir / "obs_frame.png").convert("RGB")
    depth_m = np.load(cam_dir / "obs_frame_depth.npy").astype(np.float32) / 1000.0
    candidates = sam_candidates(processor, image, prompt, conf)
    if not candidates:
        return {"status": "invalid", "reason": "no_candidates", "K": len(trajs), "K_paired": 0,
                "prompts": [prompt]}

    pairings = greedy_match(candidates, trajs, depth_m, tj["camera_info"]["intrinsics"], t2d, t3d)
    records = [{"annot_id": t.get("annot_id", ""), "status": "paired", **p} if p else
               {"annot_id": t.get("annot_id", ""), "status": "unpaired"} for t, p in zip(trajs, pairings)]
    K, K_paired = len(trajs), sum(p is not None for p in pairings)
    if K_paired < K:
        return {"status": "invalid", "reason": "unpaired_trajectories", "K": K, "K_paired": K_paired,
                "pairings": records,
                "unpaired_annot_ids": [r["annot_id"] for r in records if r["status"] == "unpaired"],
                "total_candidates": len(candidates)}
    union = np.zeros_like(candidates[0][0], dtype=bool)
    for p in pairings:
        union |= candidates[p["mask_idx"]][0]
    cv2.imwrite(str(mask_out), union.astype(np.uint8) * 255)
    return {"status": "valid", "reason": "all_paired", "K": K, "K_paired": K_paired,
            "pairings": records, "total_candidates": len(candidates)}


def run_sam_match(output_root: Path, conf: float, t2d: float, t3d: float,
                  shard: int = 0, num_shards: int = 1) -> None:
    """Process every num_shards-th cam starting at shard (one process per GPU)."""
    manifest_path = output_root / MANIFEST_NAME
    prompts = {k: v["sam3_prompt"] for k, v in load_manifest_at(output_root / QUERY_MANIFEST_NAME).items()
               if v.get("status") == "done" and isinstance(v.get("sam3_prompt"), str) and v["sam3_prompt"].strip()}
    done = set(load_manifest_at(manifest_path))
    cams = sorted((output_root / SOURCE).glob("*/interval_*/cam_*"))[shard::num_shards]
    todo = [c for c in cams if c.is_dir() and cam_key(c) not in done]
    print(f"[sam-match] shard {shard}/{num_shards}: {len(todo)} of {len(cams)} cams to process")
    if not todo:
        return
    processor = setup_sam3(conf)
    results = {}
    for i, cam_dir in enumerate(todo, start=1):
        try:
            results[cam_key(cam_dir)] = process_cam(cam_dir, prompts, processor, conf, t2d, t3d)
        except Exception as e:  # one unreadable cam must not stop a long run
            results[cam_key(cam_dir)] = {"status": "invalid", "reason": f"exception:{type(e).__name__}",
                                         "error": str(e)[:500]}
        if i % SAVE_EVERY == 0 or i == len(todo):
            save_manifest_at(manifest_path, results)
            n_valid = sum(r["status"] == "valid" for r in results.values())
            print(f"[sam-match] {i}/{len(todo)} cams, {n_valid} valid")
            results = {}
