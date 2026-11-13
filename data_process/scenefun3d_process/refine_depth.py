"""Stage 5 (run last): LingBot depth refinement of the sam-match-valid cams.

lingbot-depth (https://github.com/robbyant/lingbot-depth, installs the `mdm` package) fills
the holes of the rendered depth from RGB + depth + intrinsics. The original map is renamed to
obs_frame_depth_raw.npy and the refined one (float32 mm) is written as obs_frame_depth.npy.
Cams that have both files are skipped, so reruns resume (also after an interrupted save).
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .frame_selection import SOURCE
from .sam_match import MANIFEST_NAME

MODEL_ID = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


def load_sample(cam_dir: Path) -> dict:
    raw = cam_dir / "obs_frame_depth_raw.npy"  # exists here only if a previous save was interrupted
    depth_mm = np.load(raw if raw.exists() else cam_dir / "obs_frame_depth.npy").astype(np.float32)
    rgb = cv2.cvtColor(cv2.imread(str(cam_dir / "obs_frame.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    if depth_mm.shape != (H, W):
        depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)
    intr = json.loads((cam_dir / "trajectory.json").read_text())["camera_info"]["intrinsics"]
    K = np.array([[intr["fx"] / W, 0.0, intr["cx"] / W],
                  [0.0, intr["fy"] / H, intr["cy"] / H],
                  [0.0, 0.0, 1.0]], dtype=np.float32)  # normalized intrinsics
    return {"cam_dir": cam_dir, "rgb": rgb, "depth_mm": depth_mm, "K": K}


def refine_batch(model, samples: list, device) -> list:
    """Same-size samples -> refined depth maps (float32 mm)."""
    image = torch.from_numpy(np.stack([s["rgb"] for s in samples])).float().permute(0, 3, 1, 2) / 255.0
    depth = torch.from_numpy(np.stack([s["depth_mm"] / 1000.0 for s in samples])).float()
    K = torch.from_numpy(np.stack([s["K"] for s in samples])).float()
    with torch.no_grad():
        out = model.infer(image.to(device), depth_in=depth.to(device), intrinsics=K.to(device),
                          apply_mask=True, use_fp16=True, resolution_level=9)
    refined = out["depth"].cpu().numpy()
    if refined.ndim == 4:
        refined = refined.squeeze(1)
    return [(r * 1000.0).astype(np.float32) for r in refined]


def run_refine_depth(output_root: Path, batch_size: int, shard: int = 0, num_shards: int = 1) -> None:
    from mdm.model.v2 import MDMModel

    sam_match = json.loads((output_root / MANIFEST_NAME).read_text())
    cams = []
    for key in sorted(k for k, v in sam_match.items() if v.get("status") == "valid"):
        episode, interval_idx, cam = key.split("__")
        cams.append(output_root / SOURCE / episode / f"interval_{interval_idx}" / cam)
    cams = cams[shard::num_shards]
    todo = [c for c in cams if not ((c / "obs_frame_depth_raw.npy").exists() and (c / "obs_frame_depth.npy").exists())]
    print(f"[refine-depth] shard {shard}/{num_shards}: {len(todo)} of {len(cams)} cams to refine")
    if not todo:
        return
    device = torch.device("cuda")
    model = MDMModel.from_pretrained(MODEL_ID).to(device)

    t0, done, batch = time.time(), 0, []

    def flush():
        nonlocal done
        for s, refined in zip(batch, refine_batch(model, batch, device)):
            cam_dir = s["cam_dir"]
            np.save(cam_dir / "obs_frame_depth.tmp.npy", refined)
            if not (cam_dir / "obs_frame_depth_raw.npy").exists():
                os.rename(cam_dir / "obs_frame_depth.npy", cam_dir / "obs_frame_depth_raw.npy")
            os.replace(cam_dir / "obs_frame_depth.tmp.npy", cam_dir / "obs_frame_depth.npy")
        done += len(batch)
        print(f"[refine-depth] {done}/{len(todo)} ({done / (time.time() - t0):.1f} img/s)", flush=True)
        batch.clear()

    for cam_dir in todo:
        sample = load_sample(cam_dir)
        if batch and batch[0]["rgb"].shape != sample["rgb"].shape:
            flush()
        batch.append(sample)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
