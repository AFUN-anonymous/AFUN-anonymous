#!/usr/bin/env bash
# Preprocess VITRA Ego4D (episodes with an Ego4D fho_main contact frame).
# Only episodes in ego4d_contact_index are processed. No depth here.
set -euo pipefail

python preprocess.py vitra_ego4d \
    multi_processing=true \
    num_processes=20
