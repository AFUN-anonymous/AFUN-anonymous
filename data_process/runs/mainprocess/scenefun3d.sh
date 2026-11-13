#!/usr/bin/env bash
# Mainprocess SceneFun3D (paper Appendix C.1): no tracking; the annotated motion is
# converted directly to the Bezier target.
#   1. Frame selection, one process per visit (a laser scan needs ~28 GB RAM), then a
#      second pass that retries failed visits and merges the per-visit manifests.
#   2. contact_frame.png -> obs_frame.png (query generation reads both).
#   3. SAM3 query generation (stage 01) into a dedicated manifest. The released prompts
#      came from Qwen3.5-35B-A3B-FP8 via vLLM.
#   4. SAM3 masks paired with the trajectories (one shard per GPU).
#   5. Curve fit.
#   6. LingBot depth refinement (keeps the rendered depth as obs_frame_depth_raw.npy).
# Then: python tools/export_dataset.py {output_root} {out_dir} --datasets scenefun3d
set -euo pipefail

N_PARALLEL=${N_PARALLEL:-4}  # frame-selection processes
N_GPUS=${N_GPUS:-4}
read -r DATA_ROOT OUTPUT_ROOT < <(python -c "from omegaconf import OmegaConf; \
c = OmegaConf.load('config/paths/default.yaml'); print(c.data_root, c.output_root)")
SF3D="python -m scenefun3d_process.cli"

on_gpus() {  # run "$@" --shard i --num-shards N_GPUS on GPU i, wait for all
    local pids=()
    for i in $(seq 0 $((N_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$i "$@" --shard "$i" --num-shards "$N_GPUS" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
}

ls "${DATA_ROOT}/scenefun3d/train_val" | grep -E '^[0-9]+$' \
    | xargs -P "${N_PARALLEL}" -I{} ${SF3D} frame-selection --visits {} || true
${SF3D} frame-selection

for d in "${OUTPUT_ROOT}"/scenefun3d/*/interval_*/cam_0; do
    ln -sfn obs_frame.png "${d}/contact_frame.png"
done

python mainprocess.py stage=query_gen_vllm \
    dataset_name=scenefun3d \
    skip_existing=true \
    vlm.manifest_path="${OUTPUT_ROOT}/scenefun3d_query_gen_manifest.json"

on_gpus ${SF3D} sam-match --confidence 0.15 --proximity-2d-px 50 --proximity-3d-m 0.5
${SF3D} curve-fit
on_gpus ${SF3D} refine-depth --batch-size 16
