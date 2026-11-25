"""Tests for zero-leakage CV, Empirical-Bayes scaler, bootstrap CIs, and real OOF lift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_bdb2027.data import DatasetPaths, load_combine_results
from nfl_bdb2027.kinematics import extract_rep_level_kinematics
from nfl_bdb2027.models import (
    ANTHRO_COLS,
    OPTICAL_CORE_COLS,
    STOPWATCH_COLS,
    FoldScalerEB,
    bootstrap_metrics,
    build_cv_splits,
    run_zero_leakage_cv,
)
from nfl_bdb2027.outcomes import build_outcomes_table


def test_fold_scaler_eb_imputes_without_nans_or_leakage() -> None:
    tr = pd.DataFrame({
        "combine_position": ["WR", "WR", "WR", "OL", "OL", "OL"],
        "forty": [4.40, 4.45, np.nan, 5.10, 5.20, np.nan],
    })
    va = pd.DataFrame({
        "combine_position": ["WR", "OL", "TE"],
        "forty": [np.nan, 5.15, np.nan],
    })
    sc = FoldScalerEB(cols=["forty"], shrinkage_k=5.0).fit(tr)
    X_tr = sc.transform(tr)
    X_va = sc.transform(va)
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_va).all()
    # Missing WR in val should impute to 0.0 (its own positional shrunk mean)
    assert abs(X_va[0, 0]) < 1e-9


def test_temporal_loyo_splits_are_disjoint() -> None:
    df = pd.DataFrame({
        "draft_year": [2023, 2023, 2024, 2024, 2025, 2025],
        "combine_position": ["WR", "OL", "WR", "OL", "WR", "OL"],
    })
    splits = build_cv_splits(df, mode="temporal")
    assert len(splits) == 3
    seen_val = []
    for tr, va, _name in splits:
        assert len(set(tr) & set(va)) == 0
        assert len(set(df.iloc[tr]["draft_year"]) & set(df.iloc[va]["draft_year"])) == 0
        seen_val.extend(va.tolist())
    assert sorted(seen_val) == [0, 1, 2, 3, 4, 5]


def test_bootstrap_metrics_detects_real_lift() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    base = 0.25 * y + rng.normal(scale=0.9, size=200)
    better = 0.65 * y + rng.normal(scale=0.6, size=200)
    res = bootstrap_metrics(y, better, baseline_pred=base, n_boot=300, seed=0)
    assert res["r2"] > 0.25
    assert res["delta_r2_vs_baseline"] > 0.1
    assert res["delta_r2_p_one_sided"] < 0.02
    assert res["r2_ci95"][0] < res["r2"] < res["r2_ci95"][1]


def test_real_temporal_optical_beats_stopwatch_on_game_translation() -> None:
    """Real official data regression test: Optical kinematics beat traditional
    stopwatch+anthro on unseen future draft classes (Temporal LOYO CV)."""
    paths = DatasetPaths.from_dir()
    rep_df = extract_rep_level_kinematics(paths)
    cr_pl = load_combine_results(paths)
    outcomes_df = build_outcomes_table(paths)

    specs = {
        "stopwatch_anthro": [STOPWATCH_COLS + ANTHRO_COLS],
        "optical_only": [OPTICAL_CORE_COLS + ANTHRO_COLS],
        "stopwatch_plus_optical": [STOPWATCH_COLS + ANTHRO_COLS, OPTICAL_CORE_COLS + ANTHRO_COLS],
    }
    metrics, _, _ = run_zero_leakage_cv(
        outcomes_df,
        rep_df,
        cr_pl,
        "game_kinematic_index",
        specs,
        cv_mode="temporal",
        baseline_key="stopwatch_anthro",
        n_boot=400,
    )
    assert metrics["stopwatch_plus_optical"]["r2"] > metrics["stopwatch_anthro"]["r2"]
    assert metrics["stopwatch_plus_optical"]["spearman_rho"] > metrics["stopwatch_anthro"]["spearman_rho"]
    assert metrics["stopwatch_plus_optical"]["delta_r2_p_one_sided"] < 0.05
