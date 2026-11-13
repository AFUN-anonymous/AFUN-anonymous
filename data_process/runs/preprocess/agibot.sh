#!/usr/bin/env bash
# Preprocess AgiBot World Beta: each pre-cut action clip under `clip_root`
# becomes one interval. Depth is not read here; it comes from DA3 video
# depth in the sam3_mask_traj_depth stage.
set -euo pipefail

python preprocess.py agibot \
    multi_processing=true \
    num_processes=20
