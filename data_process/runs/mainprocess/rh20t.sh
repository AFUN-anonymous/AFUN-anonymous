#!/usr/bin/env bash
# Mainprocess RH20T.
# Sensor depth PNGs → sam3_mask_traj.
set -euo pipefail

DATASET=rh20t
python mainprocess.py stage=query_gen_vllm \
    dataset_name="${DATASET}" \
    skip_existing=true

python mainprocess.py stage=sam3_best_frame_track_concurrent \
    dataset_name="${DATASET}" \
    skip_existing=true

python mainprocess.py stage=sam3_mask_traj \
    dataset_name="${DATASET}" \
    skip_existing=true

# Fit the Bezier motion targets into each cam's trajectory.json.
OUTPUT_ROOT=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/paths/default.yaml').output_root)")
python curve_fit/run_curve_fit_dataset.py "${OUTPUT_ROOT}/${DATASET}" --skip-existing
