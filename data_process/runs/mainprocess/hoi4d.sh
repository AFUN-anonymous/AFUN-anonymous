#!/usr/bin/env bash
# Mainprocess HOI4D. No video tracking or depth back-projection: the trajectory
# comes from the HOI4D pose annotations (hoi4d_process/process_hoi4d.py).
#   1. Prepare HOI4D_OUTPUT_ROOT (= <paths.output_root>/hoi4d) for mainprocess.
#   2. VLM query generation on (obs_frame, contact_frame).
#   3. SAM3 part mask closest to the whole-object mask (hoi4d_sam3_mask.py).
set -euo pipefail

python -m hoi4d_process.hoi4d_prepare_for_mainprocess

python mainprocess.py stage=query_gen_vllm \
    dataset_name=hoi4d \
    skip_existing=true

python -m hoi4d_process.hoi4d_sam3_mask --skip-existing
