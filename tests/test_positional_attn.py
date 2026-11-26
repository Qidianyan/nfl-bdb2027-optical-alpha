"""Comprehensive unit & real-artifact verification tests for V4 Position-Specific
Multi-Scale 1D-CNN + Dual-Head Temporal Attention, BH-FDR, Partial R^2, Bootstrap 95% CI,
and Zero-Leakage Group Splits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nfl_bdb2027.config import COMBINE_POSITIONS, MATERIALS_DIR
from nfl_bdb2027.positional_attn import (
    N_ENRICHED_CHANNELS,
    POSITION_GROUPS,
    MultiScalePositionalEncoderNumpy,
    PositionalResidualMetaFusion,
    TemporalAttentionPool1D,
    benjamini_hochberg_fdr,
    bootstrap_ci_and_lift,
    build_enriched_10hz_channels,
    build_zero_leakage_group_splits,
    compute_partial_r2,
    extract_peak_saliency_window,
    fit_fold_channel_stats_v4,
    map_combine_position_to_group,
    map_pos_id_to_group,
)


def test_position_group_mapping_exhaustive_and_disjoint() -> None:
    mapped = [map_combine_position_to_group(p) for p in COMBINE_POSITIONS]
    assert set(mapped) == set(POSITION_GROUPS)
    assert len(POSITION_GROUPS) == 4
    assert map_combine_position_to_group("WR") == "WR_TE"
    assert map_combine_position_to_group("TE") == "WR_TE"
    assert map_combine_position_to_group("DB") == "DB"
    assert map_combine_position_to_group("DL") == "DL_EDGE"
    assert map_combine_position_to_group("OL") == "OL_LB"
    for pid in range(len(COMBINE_POSITIONS)):
        assert map_pos_id_to_group(pid) in POSITION_GROUPS


def test_multiscale_conv1d_and_dual_head_attention_forward_shapes() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, N_ENRICHED_CHANNELS, 48)).astype(np.float32)
    mask = np.zeros((6, 48), dtype=np.float32)
    lengths = [12, 24, 30, 48, 18, 35]
    for i, L in enumerate(lengths):
        mask[i, :L] = 1.0

    for grp in POSITION_GROUPS:
        enc = MultiScalePositionalEncoderNumpy(position_group=grp, in_channels=N_ENRICHED_CHANNELS, n_heads=2)
        assert enc.kernels == (3, 5, 9)
        out = enc.forward(x, mask)
        assert out["pred"].shape == (6,)
        assert out["embedding"].shape == (6, enc.d_model * 2)
        assert out["alpha_t"].shape == (6, 48)
        assert out["alpha_heads"].shape == (6, 2, 48)
        assert np.isfinite(out["pred"]).all()
        assert np.isfinite(out["embedding"]).all()


def test_temporal_attention_weights_sum_to_one_and_mask_padding() -> None:
    rng = np.random.default_rng(19)
    h = rng.normal(size=(5, 36, 40)).astype(np.float32)
    mask = np.zeros((5, 40), dtype=np.float32)
    lengths = [5, 11, 20, 33, 40]
    for i, L in enumerate(lengths):
        mask[i, :L] = 1.0

    for n_heads in (1, 2):
        pool = TemporalAttentionPool1D(d_model=36, n_heads=n_heads, seed=101)
        pooled, alpha, heads = pool.forward(h, mask)
        assert pooled.shape == (5, 36 * n_heads)
        assert np.isfinite(alpha).all()
        for i, L in enumerate(lengths):
            assert alpha[i, :L].sum() == pytest.approx(1.0, abs=1e-5)
            if L < 40:
                assert float(np.max(np.abs(alpha[i, L:]))) == 0.0
            for hi in range(n_heads):
                assert heads[i, hi, :L].sum() == pytest.approx(1.0, abs=1e-5)
                if L < 40:
                    assert float(np.max(np.abs(heads[i, hi, L:]))) == 0.0


def test_temporal_attention_nan_inf_and_all_zero_mask_guard() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(4, N_ENRICHED_CHANNELS, 32)).astype(np.float32)
    x[0, :, 3] = np.nan
    x[1, 0, 7] = np.inf
    x[2, 2, 10] = -np.inf

    mask = np.zeros((4, 32), dtype=np.float32)
    mask[0, :15] = 1.0
    mask[1, :20] = 1.0
    mask[2, :10] = 1.0
    # Sample 3 is completely masked (all zeros) to test empty-rep safety
    mask[3, :] = 0.0

    enc = MultiScalePositionalEncoderNumpy(position_group="WR_TE", n_heads=2)
    out = enc.forward(x, mask)
    assert np.isfinite(out["pred"]).all()
    assert np.isfinite(out["alpha_t"]).all()
    assert np.isfinite(out["alpha_heads"]).all()
    for i in (0, 1, 2):
        assert out["alpha_t"][i].sum() == pytest.approx(1.0, abs=1e-5)
    assert out["alpha_t"][3].sum() == pytest.approx(0.0, abs=1e-6)


def test_peak_0_3s_saliency_window_extraction() -> None:
    alpha = np.full(30, 0.01, dtype=np.float64)
    alpha[12:15] = [0.22, 0.30, 0.21]  # Decisive 0.3s window at 1.2s..1.5s
    alpha /= alpha.sum()
    mask = np.ones(30, dtype=np.float32)
    win = extract_peak_saliency_window(alpha, mask, window_frames=3, dt_sec=0.1)
    assert win["start_frame"] == 12
    assert win["end_frame"] == 14
    assert win["start_sec"] == pytest.approx(1.2, abs=1e-6)
    assert win["end_sec"] == pytest.approx(1.5, abs=1e-6)
    assert win["peak_frame"] == 13
    assert win["attention_mass"] > 0.70


def test_benjamini_hochberg_fdr_monotonicity_and_exact_values() -> None:
    p_vals = np.array([0.04, 0.001, 0.20, 0.01, 0.50])
    q_vals = benjamini_hochberg_fdr(p_vals)
    # Verify exact closed-form BH values: sorted p=[0.001, 0.01, 0.04, 0.20, 0.50]
    # raw q = [0.005, 0.025, 0.06667, 0.25, 0.50]
    order = np.argsort(p_vals)
    q_sorted = q_vals[order]
    assert np.all(np.diff(q_sorted) >= -1e-12)  # Strictly non-decreasing monotonicity
    assert np.all(q_vals >= p_vals - 1e-12)
    assert np.all((q_vals >= 0.0) & (q_vals <= 1.0))
    assert q_sorted[0] == pytest.approx(0.005, abs=1e-6)
    assert q_sorted[1] == pytest.approx(0.025, abs=1e-6)
    assert q_sorted[2] == pytest.approx(0.04 * 5 / 3, abs=1e-6)

    # NaN safety
    q_nan = benjamini_hochberg_fdr([0.01, float("nan"), 0.05])
    assert np.isfinite(q_nan).all()
    assert q_nan[1] == pytest.approx(1.0)


def test_partial_r2_and_bootstrap_ci_order() -> None:
    rng = np.random.default_rng(123)
    n = 240
    y = rng.normal(size=n)
    pred_red = 0.30 * y + rng.normal(scale=0.9, size=n)
    pred_full = 0.65 * y + rng.normal(scale=0.65, size=n)

    pr2 = compute_partial_r2(y, pred_red, pred_full)
    assert 0.05 < pr2 < 0.95
    assert compute_partial_r2(y, pred_red, pred_red) == pytest.approx(0.0, abs=1e-12)

    res = bootstrap_ci_and_lift(y, pred_full, baseline_pred=pred_red, n_boot=500, seed=42)
    assert res["cv_r2_ci95"][0] < res["cv_r2"] < res["cv_r2_ci95"][1]
    assert res["delta_r2_ci95"][0] < res["delta_r2"] < res["delta_r2_ci95"][1]
    assert res["partial_r2_ci95"][0] < res["partial_r2"] < res["partial_r2_ci95"][1]
    assert res["p_one_sided"] < 0.02


def test_zero_leakage_group_splits_player_and_year_isolation() -> None:
    # 1. Stratified player-disjoint splits
    nfl_ids = np.arange(1000, 1060)
    years = np.tile([2023, 2024, 2025], 20)
    positions = np.tile(["WR", "DB", "DL", "OL", "TE"], 12)
    s_splits = build_zero_leakage_group_splits(nfl_ids, years, positions, mode="stratified_player", n_splits=5, seed=42)
    assert len(s_splits) == 5
    for tr, va, _ in s_splits:
        assert len(set(nfl_ids[tr]) & set(nfl_ids[va])) == 0

    # 2. Temporal LOYO GroupKFold splits (zero player AND zero draft-year overlap)
    t_splits = build_zero_leakage_group_splits(nfl_ids, years, positions, mode="temporal_loyo")
    assert len(t_splits) == 3
    for tr, va, _ in t_splits:
        assert len(set(nfl_ids[tr]) & set(nfl_ids[va])) == 0
        assert len(set(years[tr]) & set(years[va])) == 0

    # 3. Multi-row player_group isolation
    multi_ids = np.repeat(np.arange(10, 30), 3)
    multi_yrs = np.repeat(np.tile([2023, 2024], 10), 3)
    multi_pos = np.repeat(np.tile(["WR", "DB"], 10), 3)
    g_splits = build_zero_leakage_group_splits(multi_ids, multi_yrs, multi_pos, mode="player_group", n_splits=5)
    for tr, va, _ in g_splits:
        assert len(set(multi_ids[tr]) & set(multi_ids[va])) == 0


def test_enriched_channels_and_residual_meta_fusion_no_leakage() -> None:
    rng = np.random.default_rng(77)
    seq_x = rng.normal(size=(40, 4, 5, 24)).astype(np.float32)
    seq_x[:, :, 0, :] = np.abs(seq_x[:, :, 0, :]) + 3.0
    seq_x[0, 0, 1, 5] = np.nan
    fmask = np.ones((40, 4, 24), dtype=np.float32)
    fmask[:, :, 20:] = 0.0
    rmask = np.ones((40, 4), dtype=np.float32)
    rep_len = np.full((40, 4), 20, dtype=np.int32)
    drill_id = rng.integers(0, 6, size=(40, 4))

    tr_idx = np.arange(0, 30)
    stats_d = fit_fold_channel_stats_v4(seq_x, fmask, rmask, drill_id, n_drills=6, tr_idx=tr_idx)
    x_enr = build_enriched_10hz_channels(seq_x, fmask, rep_len, drill_id, stats_d)
    assert x_enr.shape == (40, 4, N_ENRICHED_CHANNELS, 24)
    assert np.isfinite(x_enr).all()
    assert float(np.max(np.abs(x_enr[:, :, :, 20:]))) == 0.0

    # Residual meta-fusion smoke test
    y_tr = rng.normal(size=80)
    tr_blocks = np.column_stack([0.5 * y_tr + rng.normal(scale=0.6, size=80), 0.6 * y_tr + rng.normal(scale=0.5, size=80)])
    va_blocks = rng.normal(size=(20, 2))
    tr_extra = rng.normal(size=(80, 6))
    va_extra = rng.normal(size=(20, 6))
    tr_grp = np.tile(list(POSITION_GROUPS), 20)
    va_grp = np.tile(list(POSITION_GROUPS), 5)
    fuser = PositionalResidualMetaFusion()
    preds = fuser.fit_predict_fold(tr_blocks, va_blocks, tr_extra, va_extra, tr_grp, va_grp, y_tr)
    assert preds.shape == (20,)
    assert np.isfinite(preds).all()


def test_v4_gpu_metrics_and_saliency_artifacts_integrity() -> None:
    metrics_path = MATERIALS_DIR / "real_data_metrics_v4_gpu.json"
    saliency_path = MATERIALS_DIR / "v4_saliency_profiles.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    assert saliency_path.exists(), f"Missing {saliency_path}"

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(m["positional_encoders"]) >= 4
    for grp in POSITION_GROUPS:
        assert grp in m["positional_encoders"]
        pe = m["positional_encoders"][grp]
        assert pe["cv_r2_ci95"][0] <= pe["cv_r2"] <= pe["cv_r2_ci95"][1]
        assert 0.0 <= pe["q_fdr_vs_stopwatch"] <= 1.0
    assert m["overall_v4_oof"]["cv_r2"] > m["v2_baseline"]["cv_r2"]
    assert m["overall_v4_oof"]["partial_r2_vs_stopwatch"] > 0.0
    assert m["statistical_rigor"]["bootstrap_resamples"] >= 2000

    sal = json.loads(saliency_path.read_text(encoding="utf-8"))
    assert len(sal["matchups"]) >= 4
    for mu in sal["matchups"]:
        for side in ("gem", "trap"):
            prof = mu[side]
            assert prof["n_frames"] == len(prof["frames"])
            assert prof["attn_sum"] == pytest.approx(1.0, abs=1e-4)
            w_sum = sum(fr["attn_weight"] for fr in prof["frames"])
            assert w_sum == pytest.approx(1.0, abs=1e-3)
            assert all(math.isfinite(fr["attn_weight"]) for fr in prof["frames"])
