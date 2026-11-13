#!/usr/bin/env bash
# Generate RLBench_512: the 18 PerAct tasks rendered at 512x512 with RLBench's
# dataset generator (CoppeliaSim 4.1), 20 episodes per task, all variations.
# config/dataset/rlbench.yaml reads ${data_root}/RLBench_512.
#
# Needs RLBench (https://github.com/stepjam/RLBench) and CoppeliaSim. Without an
# X server, run it under a virtual display, e.g. `xvfb-run -a bash ...`.
#
# Usage:
#   COPPELIASIM_ROOT=/path/to/CoppeliaSim RLBENCH_DIR=/path/to/RLBench \
#       bash tools/generate_rlbench_512.sh /path/to/data_root/RLBench_512/data/train
set -euo pipefail

SAVE_PATH="${1:?usage: generate_rlbench_512.sh <save_path>}"
: "${COPPELIASIM_ROOT:?set COPPELIASIM_ROOT}"
: "${RLBENCH_DIR:?set RLBENCH_DIR}"

TASKS="open_drawer,push_buttons,sweep_to_dustpan_of_size,slide_block_to_color_target,meat_off_grill,turn_tap,close_jar,reach_and_drag,place_shape_in_shape_sorter,insert_onto_square_peg,put_money_in_safe,place_wine_at_rack_location,put_groceries_in_cupboard,stack_cups,light_bulb_in,put_item_in_drawer,place_cups,stack_blocks"

export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="$COPPELIASIM_ROOT"

mkdir -p "$SAVE_PATH"
python "$RLBENCH_DIR/tools/dataset_generator.py" \
    --save_path="$SAVE_PATH" \
    --tasks="$TASKS" \
    --image_size=512,512 \
    --episodes_per_task=20 \
    --renderer=opengl3 \
    --all_variations
