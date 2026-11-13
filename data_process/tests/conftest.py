import sys
from pathlib import Path

# Make the repository root importable (mainprocess, curve_fit) when running plain `pytest`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
