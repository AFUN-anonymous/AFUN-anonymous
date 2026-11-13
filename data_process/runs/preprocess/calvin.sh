#!/usr/bin/env bash
# Preprocess Calvin (640x640 re-render, training + validation splits).
# Only annotations of tasks in config/dataset/calvin_task_filter.yaml are kept.
set -euo pipefail

python preprocess.py calvin \
    multi_processing=true \
    num_processes=20
