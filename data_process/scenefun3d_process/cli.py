"""SceneFun3D -> AFUN samples. Run from the repository root; runs/mainprocess/scenefun3d.sh has the order.

    python -m scenefun3d_process.cli frame-selection [--visits all|<id>,<id>]
    python -m scenefun3d_process.cli sam-match [--shard i --num-shards n]
    python -m scenefun3d_process.cli curve-fit
    python -m scenefun3d_process.cli refine-depth [--shard i --num-shards n]

--dataset-root (raw SceneFun3D, holding train_val/) defaults to <paths.data_root>/scenefun3d and
--output-root to <paths.output_root>, both from config/paths/default.yaml.
"""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

PATHS = OmegaConf.load(Path(__file__).resolve().parents[1] / "config" / "paths" / "default.yaml")


def main(argv=None):
    p = argparse.ArgumentParser(prog="scenefun3d_process")
    p.add_argument("--dataset-root", type=Path, default=Path(PATHS.data_root) / "scenefun3d")
    p.add_argument("--output-root", type=Path, default=Path(PATHS.output_root))
    sub = p.add_subparsers(dest="stage", required=True)

    fs = sub.add_parser("frame-selection", help="write candidate cams (one process per visit is fine)")
    fs.add_argument("--visits", default="all", help="'all' (train_val visits) or comma-separated visit ids")

    sm = sub.add_parser("sam-match", help="SAM3 masks paired with the trajectories")
    sm.add_argument("--confidence", type=float, default=0.15)
    sm.add_argument("--proximity-2d-px", type=float, default=50.0)
    sm.add_argument("--proximity-3d-m", type=float, default=0.5)

    sub.add_parser("curve-fit", help="fsparams + spline_params for the valid cams")

    rd = sub.add_parser("refine-depth", help="LingBot depth refinement of the valid cams (run last)")
    rd.add_argument("--batch-size", type=int, default=16)
    for sp in (sm, rd):
        sp.add_argument("--shard", type=int, default=0)
        sp.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args(argv)

    if args.stage == "frame-selection":
        from .frame_selection import run_frame_selection
        if args.visits == "all":
            root = args.dataset_root / "train_val"
            visits = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit())
        else:
            visits = [v.strip() for v in args.visits.split(",") if v.strip()]
        run_frame_selection(args.dataset_root, args.output_root, visits)
    elif args.stage == "sam-match":
        from .sam_match import run_sam_match
        run_sam_match(args.output_root, args.confidence, args.proximity_2d_px, args.proximity_3d_m,
                      args.shard, args.num_shards)
    elif args.stage == "curve-fit":
        from .curve_fit import run_curve_fit
        run_curve_fit(args.output_root)
    elif args.stage == "refine-depth":
        from .refine_depth import run_refine_depth
        run_refine_depth(args.output_root, args.batch_size, args.shard, args.num_shards)


if __name__ == "__main__":
    main()
