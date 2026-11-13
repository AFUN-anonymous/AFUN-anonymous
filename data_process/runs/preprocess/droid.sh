#!/usr/bin/env bash
# Preprocess DROID (all episodes in the annotation patch, failures excluded).
set -euo pipefail

python preprocess.py droid \
    multi_processing=true \
    num_processes=20
