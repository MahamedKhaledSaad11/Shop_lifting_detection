"""Path setup for Phase 2: ensures Week 5 src modules are discoverable."""

import sys
from pathlib import Path

# PROJECT_ROOT is E:\Cellula_Technologies\Project3 (contains src/, Shop DataSet/, outputs/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PHASE2_ROOT is E:\Cellula_Technologies\Project3\Project3_Phase2
PHASE2_ROOT = Path(__file__).resolve().parents[1]

for path_entry in (PROJECT_ROOT, PHASE2_ROOT):
    path_str = str(path_entry)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
