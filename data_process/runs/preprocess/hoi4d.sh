#!/usr/bin/env bash
# Preprocess HOI4D.
#
# HOI4D has its own script instead of a preprocess.py adapter: it reads the
# RGB videos, 2Dseg masks, objpose JSONs and camera extrinsics, and writes
# per-interval cam_0/ directories directly. Set the paths first (see
# hoi4d_process/config.py):
#   HOI4D_ROOT, HOI4D_INSTRUCTIONS_ROOT (optional), and
#   HOI4D_OUTPUT_ROOT=<paths.output_root>/hoi4d so mainprocess finds the output.
set -euo pipefail

python -m hoi4d_process.process_hoi4d --skip-existing
