"""Master runner script for Milestone M4.

Executes:
1. phase2/evaluate_all.py (unified test evaluation, bootstrap CIs, McNemar, ROC/PR, confusion matrices)
2. phase2/benchmark.py (complexity, latency, throughput, VRAM, Pareto tradeoff plots)
3. pytest tests/ -v -s (verifies full test suite)
"""

import subprocess
import sys
import time
from pathlib import Path

# Ensure paths
PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PHASE2_ROOT, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from phase2.benchmark import run_benchmark
from phase2.evaluate_all import run_unified_evaluation


def main():
    total_start = time.time()
    print("=" * 80, flush=True)
    print("STARTING MILESTONE M4 PIPELINE", flush=True)
    print("=" * 80, flush=True)

    # 1. Run Unified Evaluation
    print("\n--- STEP 1: RUNNING UNIFIED TEST EVALUATION ---", flush=True)
    eval_results = run_unified_evaluation()

    # 2. Run Hardware Benchmark
    print("\n--- STEP 2: RUNNING HARDWARE & SPEED BENCHMARK ---", flush=True)
    bench_df = run_benchmark()

    # 3. Run Pytest Suite
    print("\n--- STEP 3: RUNNING PYTEST SUITE ---", flush=True)
    test_dir = str(PHASE2_ROOT / "tests")
    cmd = [sys.executable, "-m", "pytest", test_dir, "-v", "-s"]
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        print("\n[ERROR] Pytest failed! Check logs.", flush=True)
        sys.exit(1)
    else:
        print("\n[SUCCESS] All unit tests passed cleanly!", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"MILESTONE M4 COMPLETED IN {time.time() - total_start:.1f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
