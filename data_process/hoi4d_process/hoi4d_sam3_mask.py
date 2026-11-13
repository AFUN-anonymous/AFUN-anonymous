"""HOI4D part mask: text-prompted SAM3 on obs_frame.png.

The text prompt is the interval's `sam3_prompt` from the query_gen_vllm
manifest. Among SAM3 masks with confidence > 0.5, keep the one with the
highest confidence / (pixel distance from its centroid to the centroid of
whole_object_mask.png). Writes cam_0/sam_mask.png.

Usage:
    python -m hoi4d_process.hoi4d_sam3_mask [--skip-existing] [--limit N]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from mainprocess.utils import QUERY_GEN_MANIFEST_PATH

from .config import OUTPUT_ROOT

CONFIDENCE_THRESHOLD = 0.5


def setup_sam3(confidence_threshold: float):
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    bpe_path = os.path.join(os.path.dirname(sam3.__file__), "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def mask_centroid(mask_path: Path) -> Optional[Tuple[float, float]]:
    raw = np.array(Image.open(mask_path))
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    ys, xs = np.where(raw > 0)
    if len(ys) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def process_cam(processor, cam_dir: Path, prompt: str, contact: Optional[Tuple[float, float]],
                confidence_threshold: float) -> bool:
    """Write sam_mask.png for one cam. Returns False when no mask passes the threshold."""
    state = processor.set_image(Image.open(cam_dir / "obs_frame.png"))
    state = processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = state.get("masks"), state.get("scores")
    try:
        if masks is None or len(masks) == 0:
            return False
        masks = masks.squeeze(1).cpu().float().numpy()  # (N, H, W)
        scores = scores.cpu().float().numpy()            # (N,)

        candidates = []  # (index, confidence, centroid x, centroid y)
        for j, score in enumerate(scores):
            ys, xs = np.where(masks[j] > 0.5)
            if score > confidence_threshold and len(ys):
                candidates.append((j, float(score), float(np.mean(xs)), float(np.mean(ys))))
        if not candidates:
            return False

        if contact is not None and len(candidates) > 1:
            px, py = contact
            best = max(candidates, key=lambda c: c[1] / max(float(np.hypot(c[2] - px, c[3] - py)), 1.0))
        else:
            best = max(candidates, key=lambda c: c[1])

        mask = masks[best[0]]
        Image.fromarray(mask.astype(np.uint8) * 255).save(str(cam_dir / "sam_mask.png"))
        return True
    finally:
        processor.reset_all_prompts(state)


def main():
    parser = argparse.ArgumentParser(description="HOI4D SAM3 part mask")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--qg-manifest", type=Path, default=QUERY_GEN_MANIFEST_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # query_gen manifest keys are "<episode_id>/<interval>"; HOI4D episode ids contain "hoi4d".
    prompts = {
        k: v["sam3_prompt"]
        for k, v in json.loads(args.qg_manifest.read_text()).items()
        if "hoi4d" in k and v.get("status") == "done" and v.get("sam3_prompt")
    }

    items = []
    for iv_dir in sorted(args.output_root.glob("*/interval_*")):
        prompt = prompts.get(f"{iv_dir.parent.name}/{iv_dir.name}")
        cam_dir = iv_dir / "cam_0"
        if not prompt or not (cam_dir / "obs_frame.png").exists():
            continue
        if args.skip_existing and (cam_dir / "sam_mask.png").exists():
            continue
        whole_mask = cam_dir / "whole_object_mask.png"
        items.append((cam_dir, prompt, mask_centroid(whole_mask) if whole_mask.exists() else None))
        if args.limit and len(items) >= args.limit:
            break
    print(f"{len(prompts)} HOI4D prompts, {len(items)} cams to process")
    if not items:
        return

    processor = setup_sam3(args.confidence)
    kept = failed = 0
    for cam_dir, prompt, contact in items:
        try:
            kept += process_cam(processor, cam_dir, prompt, contact, args.confidence)
        except Exception as exc:  # keep going; one bad frame should not stop the run
            failed += 1
            print(f"ERROR {cam_dir}: {exc}")
    print(f"Done: {kept} masks, {len(items) - kept - failed} without a mask, {failed} errors")


if __name__ == "__main__":
    main()
