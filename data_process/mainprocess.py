"""
Mainprocess entry point: turns preprocess output into SAM3 queries, masks,
tracks and 3D object trajectories. All stages share config/mainprocess.yaml.

Stages (file prefix in mainprocess/):
    query_gen_vllm                    01    SAM3 text query from instruction + frames (Qwen3.5 via vLLM)
    sam3_best_frame_track_concurrent  02_5  SAM3 best-frame scan + video tracking
    sam3_mask_traj                    04    3D trajectory from tracked masks + sensor depth
    sam3_mask_traj_depth              04    3D trajectory from tracked masks + DA3 video depth

Usage:
    python mainprocess.py droid stage=query_gen_vllm       # positional dataset name
    python mainprocess.py stage=sam3_mask_traj dataset_name=droid skip_existing=true limit=5
    python mainprocess.py dataset_name=droid                # default pipeline (see _DEFAULT_PIPELINE)

See runs/mainprocess/<dataset>.sh for the exact commands used per dataset.
"""

import argparse
import importlib
import json
import os
import sys
import time

from hydra import compose, initialize
from omegaconf import DictConfig, ListConfig, OmegaConf

from mainprocess.utils import PATHS_ENV

_STAGES = {
    "query_gen_vllm": ("01_query_gen_vllm", "run_query_gen_vllm"),
    "sam3_best_frame_track_concurrent": (
        "02_5_sam3_best_frame_tracking_concurrent", "run_sam3_best_frame_track_concurrent",
    ),
    "sam3_mask_traj": ("04_sam3_mask_traj", "run_sam3_mask_traj"),
    "sam3_mask_traj_depth": ("04_sam3_mask_traj_depth", "run_sam3_mask_traj_depth"),
}

# Run when `stage` is not given (datasets with sensor depth).
_DEFAULT_PIPELINE = ["query_gen_vllm", "sam3_best_frame_track_concurrent", "sam3_mask_traj"]


def _parse_cli_args(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(description="Mainprocess pipeline entry point.")
    parser.add_argument(
        "datasetname",
        type=str,
        default=None,
        nargs="?",
        help="Optional dataset name; same as dataset_name=<name>.",
    )
    args, overrides = parser.parse_known_args(argv)

    datasetname = args.datasetname
    # Keep Hydra-style first-token overrides (e.g., stage=query_gen_vllm) working.
    if datasetname and ("=" in datasetname or datasetname.startswith(("+", "~"))):
        overrides.insert(0, datasetname)
        datasetname = None

    if datasetname is not None:
        overrides.append(f"dataset_name={datasetname}")

    return overrides


def _run_stage(cfg: DictConfig, stage: str):
    """Import and run a single stage."""
    mod_name, func_name = _STAGES[stage]
    mod = importlib.import_module(f"mainprocess.{mod_name}")
    getattr(mod, func_name)(cfg)


def main(cfg: DictConfig):
    stage = cfg.get("stage")

    if stage is None:
        print("=" * 70)
        print("No stage specified: running full pipeline:")
        print(f"  {' -> '.join(_DEFAULT_PIPELINE)}")
        print("=" * 70)
        t0 = time.time()
        for i, s in enumerate(_DEFAULT_PIPELINE, 1):
            print(f"# Pipeline step {i}/{len(_DEFAULT_PIPELINE)}: {s}")
            _run_stage(cfg, s)
        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"Full pipeline completed in {elapsed:.1f}s.")
        print(f"{'=' * 70}")
        return

    stages = stage if isinstance(stage, (list, ListConfig)) else [stage]
    for s in stages:
        if s not in _STAGES:
            raise ValueError(
                f"Unknown stage '{s}'. Choose from: {list(_STAGES)}"
            )
        _run_stage(cfg, s)


if __name__ == "__main__":
    forwarded_overrides = _parse_cli_args(sys.argv[1:])
    with initialize(config_path="config", version_base=None):
        cfg: DictConfig = compose(config_name="mainprocess", overrides=forwarded_overrides)
    # Stages rebuild dataset loaders from config/paths/default.yaml; pass them the
    # composed paths so command-line overrides (paths.data_root=...) apply there too.
    os.environ[PATHS_ENV] = json.dumps(OmegaConf.to_container(cfg.paths, resolve=True))
    main(cfg)
