#!/usr/bin/env bash
# Preprocess RH20T (robot scenes).
# Reads the task list (task_txt_path) and subset list from config/dataset/rh20t.yaml.
set -euo pipefail

python preprocess.py rh20t \
    multi_processing=true \
    num_processes=20
