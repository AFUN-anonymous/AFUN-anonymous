"""Download AFUN weights and cache the pretrained backbones."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_REPOSITORY = "afun-anonymous/AFUN"  # model repo; a dataset repo has the same name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones-only", action="store_true", help="Skip the trained AFUN checkpoint.")
    parser.add_argument("--checkpoint-only", action="store_true", help="Skip pretrained backbones.")
    parser.add_argument("--include-depth", action="store_true", help="Also cache optional RGB/sensor depth models.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/afun.pt")
    args = parser.parse_args()
    if args.backbones_only and args.checkpoint_only:
        parser.error("Choose --backbones-only or --checkpoint-only.")
    if not args.backbones_only and not args.checkpoint.is_file():
        path = hf_hub_download(CHECKPOINT_REPOSITORY, "afun.pt", repo_type="model", local_dir=args.checkpoint.parent)
        Path(path).replace(args.checkpoint)
    if not args.checkpoint_only:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="inference")
        for repository in (cfg.llm.model_name, cfg.decoder.depth_encoder.ckpt_path):
            snapshot_download(repository)
        # SAM3 is gated; accept its model terms and run `hf auth login` first.
        snapshot_download("facebook/sam3", allow_patterns=["sam3.pt", "config.json"])
        if args.include_depth:
            for repository in ("depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                               "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"):
                snapshot_download(repository)
    print("Weights are ready.")


if __name__ == "__main__":
    main()
