#!/usr/bin/env bash
# Preprocess RoboMIND V1.
# Discovers HDF5 trajectories under benchmark1_0_compressed/extracted,
# uses per-step language annotations.
set -euo pipefail

python preprocess.py robomind \
    multi_processing=true \
    num_processes=20
