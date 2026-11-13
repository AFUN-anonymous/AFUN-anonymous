#!/usr/bin/env bash
# Preprocess RLBench.
# Discovers all tasks under data/train/; 4 static cameras (wrist excluded).
set -euo pipefail

python preprocess.py rlbench \
    multi_processing=true \
    num_processes=20
