#!/usr/bin/env bash
# Mainprocess VITRA EPIC-Kitchens.
# Same architecture as vitra_ego4d_noannot but with the standard (annotated)
# vitra VLM template and the dedicated epic query_gen manifest.
set -euo pipefail

DATASET=vitra_epic
QG_MANIFEST=query_gen_cache/query_gen_manifest_vitra_epic.json
VLM_CFG=mainprocess/vlm_captioning/configs/sam3_query_gen_vitra.yaml

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
