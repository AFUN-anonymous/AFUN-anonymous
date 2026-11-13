"""HOI4D processing configuration -- paths, constants, category mappings.

Paths come from environment variables:
  HOI4D_ROOT               HOI4D download (layout below)
  HOI4D_INSTRUCTIONS_ROOT  clone of HOI4D-Instructions (release.txt, definitions/);
                           default: $HOI4D_ROOT/HOI4D-Instructions
  HOI4D_OUTPUT_ROOT        output dir; use <paths.output_root>/hoi4d so that
                           mainprocess (dataset_name=hoi4d) finds it
"""

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================

HOI4D_ROOT = Path(os.environ.get("HOI4D_ROOT", "/path/to/HOI4D"))
RGB_ROOT = HOI4D_ROOT / "HOI4D_release"
ANNO_ROOT = HOI4D_ROOT / "HOI4D_annotations"
DEPTH_ROOT = HOI4D_ROOT / "depth_download" / "HOI4D_depth_video"
CAMERA_ROOT = HOI4D_ROOT / "camera_params" / "camera_params"
REPO_ROOT = Path(os.environ.get("HOI4D_INSTRUCTIONS_ROOT", HOI4D_ROOT / "HOI4D-Instructions"))
OUTPUT_ROOT = Path(os.environ.get("HOI4D_OUTPUT_ROOT", "/path/to/output_root/hoi4d"))

RELEASE_TXT = REPO_ROOT / "release.txt"
TASK_DEFS_CSV = REPO_ROOT / "definitions" / "task" / "task_definitions.csv"

# ============================================================
# HOI4D constants
# ============================================================

FPS = 15
IMAGE_H, IMAGE_W = 1080, 1920

# ============================================================
# Category mapping
# ============================================================

CATEGORIES = {
    "C1": "ToyCar", "C2": "Mug", "C3": "Laptop", "C4": "StorageFurniture",
    "C5": "Bottle", "C6": "Safe", "C7": "Bowl", "C8": "Bucket",
    "C9": "Scissors", "C11": "Pliers", "C12": "Kettle", "C13": "Knife",
    "C14": "TrashCan", "C17": "Lamp", "C18": "Stapler", "C20": "Chair",
}

# ============================================================
# Motion filter
# ============================================================

MIN_WORLD_DISPLACEMENT_M = 0.03  # 3cm

# ============================================================
# Action event grouping
# ============================================================

MANIPULATION_START = {"reachout", "reach"}
IDLE_EVENTS = {"rest", "stop", "idle"}
