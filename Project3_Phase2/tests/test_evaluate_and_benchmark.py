"""Unit tests verifying that unified evaluation and hardware benchmark produce valid artifacts."""

import json
from pathlib import Path
import pandas as pd
import pytest

from phase2.config import METRICS_DIR, PLOTS_DIR


def test_metrics_all_artifacts_exist_and_valid():
    """Verify metrics_all.json exists and contains complete valid metrics for all models."""
    json_path = METRICS_DIR / "metrics_all.json"
    assert json_path.exists(), f"Missing {json_path}"

    with open(json_path, "r") as f:
        data = json.load(f)

    assert "models" in data
    assert "mcnemar_tests" in data
    assert "comparison_table" in data

    expected_models = [
        "scratch_cnn3d",
        "scratch_cnn_rnn",
        "pretrained_cnn_rnn",
        "pretrained_cnn3d",
        "videomae_small",
        "videomae_base",
    ]

    for m in expected_models:
        assert m in data["models"], f"Model {m} missing from metrics_all.json"
        m_info = data["models"][m]
        assert "test_single_tau05" in m_info
        assert "test_tta_tau05" in m_info
        assert "bootstrap_ci95" in m_info

        f1 = m_info["test_single_tau05"]["f1_macro"]
        acc = m_info["test_single_tau05"]["accuracy"]
        assert 0.80 <= f1 <= 1.0, f"Unreasonable F1 score for {m}: {f1}"
        assert 0.80 <= acc <= 1.0, f"Unreasonable accuracy for {m}: {acc}"

    # Verify Fine-tuned HF models outperform scratch models
    vmae_b_f1 = data["models"]["videomae_base"]["test_single_tau05"]["f1_macro"]
    vmae_s_f1 = data["models"]["videomae_small"]["test_single_tau05"]["f1_macro"]
    scratch_3d_f1 = data["models"]["scratch_cnn3d"]["test_single_tau05"]["f1_macro"]
    scratch_rnn_f1 = data["models"]["scratch_cnn_rnn"]["test_single_tau05"]["f1_macro"]

    assert vmae_b_f1 > scratch_3d_f1
    assert vmae_b_f1 > scratch_rnn_f1
    assert vmae_s_f1 > scratch_3d_f1
    assert vmae_s_f1 > scratch_rnn_f1


def test_benchmark_artifacts_exist_and_valid():
    """Verify benchmark.csv and benchmark.json exist and contain valid measurements."""
    csv_path = METRICS_DIR / "benchmark.csv"
    json_path = METRICS_DIR / "benchmark.json"

    assert csv_path.exists(), f"Missing {csv_path}"
    assert json_path.exists(), f"Missing {json_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 6, f"Expected 6 benchmark rows, got {len(df)}"

    required_cols = [
        "model_key", "params_m", "checkpoint_disk_mb", "gflops_per_clip",
        "latency_fp16_ms", "throughput_b1_clips_sec", "total_e2e_latency_ms"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing benchmark column: {col}"
        assert not df[col].isna().any(), f"NaN values detected in {col}"


def test_plot_artifacts_exist():
    """Verify all 4 required visualization plots exist and are non-empty."""
    required_plots = [
        "confusion_matrices_all.png",
        "roc_pr_all.png",
        "f1_accuracy_comparison.png",
        "pareto_tradeoff.png",
    ]
    for p_name in required_plots:
        p_path = PLOTS_DIR / p_name
        assert p_path.exists(), f"Missing plot: {p_path}"
        assert p_path.stat().st_size > 5000, f"Plot {p_path} appears empty or corrupt"
