#!/usr/bin/env bash
# Preprocess RH20T human-demonstration scenes (human_only=true): one interval
# per scene with sensor depth, no robot trajectory.
# Reads the task list (task_txt_path) and subset list from config/dataset/rh20t_human.yaml.
set -euo pipefail

python preprocess.py rh20t_human \
    multi_processing=true \
    num_processes=20
