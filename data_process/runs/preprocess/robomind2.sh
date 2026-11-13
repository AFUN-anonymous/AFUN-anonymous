#!/usr/bin/env bash
# Preprocess RoboMIND2.
# Top-level adapter auto-dispatches franka / ark / tienkung / ur5 by path.
set -euo pipefail

python preprocess.py robomind2 \
    multi_processing=true \
    num_processes=20
