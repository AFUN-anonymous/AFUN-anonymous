#!/usr/bin/env bash
# Preprocess VITRA Ego4D without contact annotations (episodes from
# episode_txt_list). Also saves the middle and end frames of each interval
# (query_aux_frame_mid/end.png) for the 3-frame sam3_query_gen_egonoannot prompt.
set -euo pipefail

python preprocess.py vitra_ego4d_noannot \
    multi_processing=true \
    num_processes=20
