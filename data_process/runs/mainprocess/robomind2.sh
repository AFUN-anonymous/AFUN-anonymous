#!/usr/bin/env bash
# Mainprocess RoboMIND2.
# Uses DA3 video depth (sam3_mask_traj_depth) and a dedicated query_gen
# manifest + VLM config so it does not collide with other datasets.
set -euo pipefail

DATASET=robomind2
QG_MANIFEST=query_gen_cache/query_gen_manifest_robomind2.json
VLM_CFG=mainprocess/vlm_captioning/configs/sam3_query_gen_robomind2.yaml

python mainprocess.py stage=query_gen_vllm \
    dataset_name="${DATASET}" \
    skip_existing=true \
    vlm.manifest_path="${QG_MANIFEST}" \
    vlm.config_path="${VLM_CFG}"

python mainprocess.py stage=sam3_best_frame_track_concurrent \
    dataset_name="${DATASET}" \
    skip_existing=true \
    sam3_best_frame_track.qg_manifest_path="${QG_MANIFEST}"

python mainprocess.py stage=sam3_mask_traj_depth \
    dataset_name="${DATASET}" \
    skip_existing=true

# Fit the Bezier motion targets into each cam's trajectory.json.
OUTPUT_ROOT=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/paths/default.yaml').output_root)")
python curve_fit/run_curve_fit_dataset.py "${OUTPUT_ROOT}/${DATASET}" --skip-existing
