"""Zero-leakage temporal & stratified validation, Empirical-Bayes positional
shrinkage, LightGBM+XGBoost+Ridge modality stacking, Colab GPU PyTorch 1D-CNN +
Attention deep fusion, and paired bootstrap statistical inference."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from scipy import optimize, stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from .config import ARTIFACTS_DIR, COMBINE_POSITIONS
from .kinematics import aggregate_player_kinematics

warnings.filterwarnings("ignore", category=UserWarning)

ANTHRO_COLS = [
    "combine_height",
    "combine_weight",
    "hand_size",
    "arm_length",
    "wing_span",
]

STOPWATCH_COLS = [
    "ten_yd_split",
    "forty",
    "vertical",
    "broad_jump",
    "three_cone",
    "short_shuttle",
    "bench_reps",
]

NGS_COLS = [
    "ngs_athleticism_score",
    "ngs_college_production_score",
    "ngs_final_score",
]

OPTICAL_CORE_COLS = [
    "opt_sk_max_s",
    "opt_sk_max_a",
    "opt_power_index",
    "opt_speed_index",
    "opt_burst_index",
    "opt_cbi_index",
    "opt_omega_z",
    "opt_acent_z",
    "opt_decel_z",
    "opt_reaccel_z",
    "opt_turn_apex_z",
    "opt_rep_consistency_sd",
    "opt_impulse_momentum",
    "opt_mass_power",
    "opt_braking_force",
    "opt_f40_max_s",
    "opt_f40_max_a",
    "opt_f40_first1s_a",
    "opt_f40_s_at_1s",
    "opt_f40_vel_ret",
    "opt_f40_lat_drift",
    "opt_f40_max_power",
]

DRAFT_COLS = ["log_pick"]
CNN_RAW_BLOCK_KEY = "__CNN_1D_ATTENTION_RAW__"
CNN_MULTI_BLOCK_KEY = "__CNN_1D_ATTENTION_MULTI__"
CNN_HYBRID_OPTICAL_BLOCK_KEY = "__CNN_HYBRID_OPTICAL_GBDT__"


@dataclass
class FoldScalerEB:
    """Within-fold Empirical-Bayes positional imputer and z-score normalizer."""

    cols: list[str]
    shrinkage_k: float = 10.0

    def fit(self, df: pd.DataFrame) -> "FoldScalerEB":
        self.global_mean_: dict[str, float] = {}
        self.global_std_: dict[str, float] = {}
        self.pos_mean_: dict[str, dict[str, float]] = {}
        self.pos_std_: dict[str, dict[str, float]] = {}

        pos_s = df["combine_position"]
        for c in self.cols:
            vals = df[c].dropna()
            gm = float(vals.mean()) if len(vals) > 0 else 0.0
            gs = float(vals.std()) if len(vals) > 1 and vals.std() > 1e-6 else 1.0
            self.global_mean_[c] = gm
            self.global_std_[c] = gs
            pm: dict[str, float] = {}
            ps: dict[str, float] = {}
            for p in COMBINE_POSITIONS:
                pv = df.loc[pos_s == p, c].dropna()
                n_p = float(len(pv))
                if n_p > 0:
                    m_raw = float(pv.mean())
                    s_raw = float(pv.std()) if n_p > 1 and pv.std() > 1e-6 else gs
                    pm[p] = (n_p * m_raw + self.shrinkage_k * gm) / (n_p + self.shrinkage_k)
                    ps[p] = max((n_p * s_raw + self.shrinkage_k * gs) / (n_p + self.shrinkage_k), 1e-4)
                else:
                    pm[p] = gm
                    ps[p] = gs
            self.pos_mean_[c] = pm
            self.pos_std_[c] = ps
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        pos_s = df["combine_position"]
        n = len(df)
        mat = np.zeros((n, len(self.cols)), dtype=np.float64)
        for j, c in enumerate(self.cols):
            gm = self.global_mean_[c]
            gs = self.global_std_[c]
            pm = pos_s.map(self.pos_mean_[c]).fillna(gm).to_numpy(dtype=np.float64)
            ps = pos_s.map(self.pos_std_[c]).fillna(gs).to_numpy(dtype=np.float64)
            raw = df[c].to_numpy(dtype=np.float64)
            filled = np.where(np.isnan(raw), pm, raw)
            mat[:, j] = np.clip((filled - pm) / ps, -4.5, 4.5)
        return mat


def load_colab_cnn_artifacts(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load Colab GPU PyTorch 1D-CNN + Attention nested OOF bundle & run metadata."""
    oof_path = Path(artifacts_dir) / "colab_temporal_cnn_oof.npz"
    meta_path = Path(artifacts_dir) / "colab_gpu_run_metadata.json"
    if not oof_path.exists():
        return None, None
    npz = dict(np.load(oof_path, allow_pickle=True))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return npz, meta


def _fit_predict_modality(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    cols: list[str],
    y_tr: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Fit RidgeCV + LightGBM + XGBoost ensemble on one feature modality using
    strictly within-fold Empirical-Bayes positional standardization."""
    scaler = FoldScalerEB(cols=cols, shrinkage_k=10.0).fit(tr_df)
    X_tr = scaler.transform(tr_df)
    X_va = scaler.transform(va_df)

    ridge = RidgeCV(alphas=np.logspace(0.5, 4.0, 40))
    ridge.fit(X_tr, y_tr)
    p_ridge = ridge.predict(X_va)

    m_lgb = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=2.5,
        reg_lambda=12.0,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )
    m_lgb.fit(X_tr, y_tr)
    p_lgb = m_lgb.predict(X_va)

    m_xgb = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=22,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=2.5,
        reg_lambda=12.0,
        n_jobs=1,
        tree_method="hist",
        random_state=seed,
        verbosity=0,
    )
    m_xgb.fit(X_tr, y_tr)
    p_xgb = m_xgb.predict(X_va)

    return 0.60 * p_ridge + 0.22 * p_lgb + 0.18 * p_xgb


def _fit_nonneg_meta(Z_in: np.ndarray, y_z: np.ndarray, Z_va: np.ndarray, seed: int = 42) -> np.ndarray:
    """Fit exact non-negative L2-regularized meta-stacker choosing lambda via inner CV."""
    n_tr, n_mod = Z_in.shape
    alphas = [0.02, 0.08, 0.25, 0.8, 2.5]
    kf = KFold(n_splits=min(5, max(2, n_tr // 20)), shuffle=True, random_state=seed)

    def _solve(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        m = X.shape[1]

        def obj(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:m]
            b = params[m]
            err = X @ w + b - y
            loss = float(np.mean(err * err)) + lam * float(np.sum(w * w))
            grad_w = (2.0 / len(y)) * (X.T @ err) + 2.0 * lam * w
            grad_b = float(2.0 * np.mean(err))
            return loss, np.append(grad_w, grad_b)

        init = np.append(np.full(m, 1.0 / max(m, 1)), 0.0)
        bounds = [(0.0, None)] * m + [(None, None)]
        res = optimize.minimize(obj, init, jac=True, method="L-BFGS-B", bounds=bounds)
        return res.x[:m], float(res.x[m])

    best_lam = alphas[1]
    best_mse = float("inf")
    for lam in alphas:
        oof = np.zeros(n_tr, dtype=np.float64)
        for itr, iva in kf.split(Z_in):
            w_cv, b_cv = _solve(Z_in[itr], y_z[itr], lam)
            oof[iva] = Z_in[iva] @ w_cv + b_cv
        mse = float(np.mean((oof - y_z) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_lam = lam

    w_full, b_full = _solve(Z_in, y_z, best_lam)
    if w_full.sum() <= 1e-6:
        return Z_va @ (np.ones(n_mod, dtype=np.float64) / n_mod)
    return b_full + Z_va @ w_full


def _fit_predict_stacked_modalities(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    modality_blocks: list[list[str]],
    y_tr: np.ndarray,
    seed: int = 42,
    block_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] | None = None,
    cnn_fold_data: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Zero-leakage Inner-CV Modality Meta-Stacking with per-fold modality caching
    and native support for Colab GPU PyTorch 1D-CNN + Attention nested OOF blocks."""
    n_tr = len(tr_df)
    n_mod = len(modality_blocks)

    tr_reset = tr_df.reset_index(drop=True)
    va_reset = va_df.reset_index(drop=True)
    pos_counts = tr_reset["combine_position"].value_counts()
    if len(pos_counts) > 1 and int(pos_counts.min()) >= 5:
        inner_splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 7).split(
                tr_reset, tr_reset["combine_position"]
            )
        )
    else:
        inner_splits = list(KFold(n_splits=5, shuffle=True, random_state=seed + 7).split(tr_reset))

    def _eval_block(cols: list[str], m_idx: int) -> tuple[np.ndarray, np.ndarray]:
        bkey = tuple(cols)
        if block_cache is not None and bkey in block_cache:
            return block_cache[bkey]

        if cols == [CNN_RAW_BLOCK_KEY]:
            if cnn_fold_data is None:
                raise RuntimeError("CNN block requested but cnn_fold_data is None.")
            cnn_in, cnn_va = cnn_fold_data
            m_inner = cnn_in[:, 0].astype(np.float64)
            m_va = cnn_va[:, 0].astype(np.float64)
        elif cols == [CNN_MULTI_BLOCK_KEY]:
            if cnn_fold_data is None:
                raise RuntimeError("CNN multi-head block requested but cnn_fold_data is None.")
            cnn_in, cnn_va = cnn_fold_data
            # Strictly fit strongly regularized RidgeCV on multi-task semantic heads inside outer tr_df
            m_inner = np.zeros(n_tr, dtype=np.float64)
            c_mu = np.mean(cnn_in, axis=0, keepdims=True)
            c_sd = np.maximum(np.std(cnn_in, axis=0, keepdims=True), 1e-4)
            Z_c_tr = (cnn_in - c_mu) / c_sd
            Z_c_va = (cnn_va - c_mu) / c_sd
            for itr, iva in inner_splits:
                reg = RidgeCV(alphas=np.logspace(1.5, 4.5, 30))
                reg.fit(Z_c_tr[itr], y_tr[itr])
                m_inner[iva] = reg.predict(Z_c_tr[iva])
            reg_full = RidgeCV(alphas=np.logspace(1.5, 4.5, 30))
            reg_full.fit(Z_c_tr, y_tr)
            m_va = reg_full.predict(Z_c_va)
        elif cols == [CNN_HYBRID_OPTICAL_BLOCK_KEY]:
            if cnn_fold_data is None:
                raise RuntimeError("Hybrid CNN+Optical block requested but cnn_fold_data is None.")
            cnn_in, cnn_va = cnn_fold_data
            cnn_cols = [f"__cnn_sem_{j}" for j in range(cnn_in.shape[1])]
            tr_aug = tr_reset.copy()
            va_aug = va_reset.copy()
            for j, cname in enumerate(cnn_cols):
                tr_aug[cname] = cnn_in[:, j].astype(np.float64)
                va_aug[cname] = cnn_va[:, j].astype(np.float64)
            aug_cols = OPTICAL_CORE_COLS + ANTHRO_COLS + cnn_cols
            m_inner = np.zeros(n_tr, dtype=np.float64)
            for itr, iva in inner_splits:
                m_inner[iva] = _fit_predict_modality(
                    tr_aug.iloc[itr], tr_aug.iloc[iva], aug_cols, y_tr[itr], seed=seed + m_idx
                )
            m_va = _fit_predict_modality(tr_aug, va_aug, aug_cols, y_tr, seed=seed + m_idx)
        else:
            m_inner = np.zeros(n_tr, dtype=np.float64)
            for itr, iva in inner_splits:
                m_inner[iva] = _fit_predict_modality(
                    tr_reset.iloc[itr], tr_reset.iloc[iva], cols, y_tr[itr], seed=seed + m_idx
                )
            m_va = _fit_predict_modality(tr_reset, va_reset, cols, y_tr, seed=seed + m_idx)

        if block_cache is not None:
            block_cache[bkey] = (m_inner, m_va)
        return m_inner, m_va

    if n_mod == 1:
        _, va_p = _eval_block(modality_blocks[0], 0)
        return va_p

    inner_oof = np.zeros((n_tr, n_mod), dtype=np.float64)
    va_preds = np.zeros((len(va_reset), n_mod), dtype=np.float64)
    for m_idx, cols in enumerate(modality_blocks):
        inner_oof[:, m_idx], va_preds[:, m_idx] = _eval_block(cols, m_idx)

    y_mean = float(np.mean(y_tr))
    y_std = max(float(np.std(y_tr)), 1e-6)
    Z_in = (inner_oof - y_mean) / y_std
    Z_va = (va_preds - y_mean) / y_std
    y_z = (y_tr - y_mean) / y_std

    pred_z = _fit_nonneg_meta(Z_in, y_z, Z_va, seed=seed + 19)
    return y_mean + pred_z * y_std


def build_cv_splits(
    df: pd.DataFrame,
    mode: str = "stratified",
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Create strict cross-validation splits:
    - 'stratified': 5-fold StratifiedKFold balanced on combine_position
    - 'temporal': 3-fold Leave-One-Draft-Year-Out across 2023, 2024, 2025
    """
    if mode == "stratified":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return [
            (tr, va, f"fold_{i + 1}")
            for i, (tr, va) in enumerate(skf.split(df, df["combine_position"]))
        ]
    if mode == "temporal":
        splits = []
        for yr in sorted(df["draft_year"].unique()):
            tr = np.where(df["draft_year"].to_numpy() != yr)[0]
            va = np.where(df["draft_year"].to_numpy() == yr)[0]
            splits.append((tr, va, f"holdout_{int(yr)}"))
        return splits
    raise ValueError(f"Unsupported CV mode: {mode}")


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute exact OOF metrics, 95% bootstrap CIs, and paired lift significance."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    r2 = float(r2_score(y_true, y_pred))
    rho, rho_p = stats.spearmanr(y_true, y_pred)
    rmse = float(root_mean_squared_error(y_true, y_pred))

    boot_r2 = np.empty(n_boot, dtype=np.float64)
    boot_rho = np.empty(n_boot, dtype=np.float64)
    boot_rmse = np.empty(n_boot, dtype=np.float64)
    boot_dr2 = np.empty(n_boot, dtype=np.float64) if baseline_pred is not None else None
    boot_drho = np.empty(n_boot, dtype=np.float64) if baseline_pred is not None else None

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        boot_r2[b] = r2_score(yt, yp)
        boot_rho[b] = stats.spearmanr(yt, yp)[0]
        boot_rmse[b] = root_mean_squared_error(yt, yp)
        if baseline_pred is not None:
            yb = baseline_pred[idx]
            boot_dr2[b] = boot_r2[b] - r2_score(yt, yb)
            boot_drho[b] = boot_rho[b] - stats.spearmanr(yt, yb)[0]

    res: dict[str, Any] = {
        "n": int(n),
        "r2": r2,
        "r2_ci95": [float(np.nanpercentile(boot_r2, 2.5)), float(np.nanpercentile(boot_r2, 97.5))],
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "rho_ci95": [float(np.nanpercentile(boot_rho, 2.5)), float(np.nanpercentile(boot_rho, 97.5))],
        "rmse": rmse,
        "rmse_ci95": [float(np.nanpercentile(boot_rmse, 2.5)), float(np.nanpercentile(boot_rmse, 97.5))],
    }

    if baseline_pred is not None and boot_dr2 is not None and boot_drho is not None:
        base_r2 = float(r2_score(y_true, baseline_pred))
        base_rho = float(stats.spearmanr(y_true, baseline_pred)[0])
        res["delta_r2_vs_baseline"] = r2 - base_r2
        res["delta_r2_ci95"] = [
            float(np.nanpercentile(boot_dr2, 2.5)),
            float(np.nanpercentile(boot_dr2, 97.5)),
        ]
        res["delta_r2_p_one_sided"] = float(np.mean(boot_dr2 <= 0.0))
        res["delta_rho_vs_baseline"] = float(rho) - base_rho
        res["delta_rho_ci95"] = [
            float(np.nanpercentile(boot_drho, 2.5)),
            float(np.nanpercentile(boot_drho, 97.5)),
        ]
        res["delta_rho_p_one_sided"] = float(np.mean(boot_drho <= 0.0))
    return res


def run_zero_leakage_cv(
    outcomes_df: pd.DataFrame,
    rep_df: pl.DataFrame,
    combine_results_pl: pl.DataFrame,
    target_col: str,
    model_specs: dict[str, list[list[str]]],
    cv_mode: str = "stratified",
    baseline_key: str = "stopwatch_anthro",
    n_boot: int = 2000,
    seed: int = 42,
    cnn_npz: dict[str, Any] | None = None,
    cnn_tag: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], pd.DataFrame]:
    """Run strictly zero-leakage CV where within-drill optical standardization,
    Empirical-Bayes positional scaling, base learners, PyTorch 1D-CNN inner OOF
    representations, and modality meta-weights are ALL fit strictly inside each
    outer training fold."""
    sub_outcomes = (
        outcomes_df[outcomes_df[target_col].notna()]
        .sort_values("nfl_id")
        .reset_index(drop=True)
    )
    y = sub_outcomes[target_col].to_numpy(dtype=np.float64)
    splits = build_cv_splits(sub_outcomes, mode=cv_mode, seed=seed)

    oof_preds: dict[str, np.ndarray] = {
        k: np.zeros(len(sub_outcomes), dtype=np.float64) for k in model_specs
    }

    for fold_i, (tr_idx, va_idx, _fold_name) in enumerate(splits):
        fit_ids = sub_outcomes.iloc[tr_idx]["nfl_id"].tolist()
        kin_fold_pd = aggregate_player_kinematics(
            rep_df, combine_results_pl, fit_nfl_ids=fit_ids
        ).to_pandas()
        fold_df = sub_outcomes.merge(kin_fold_pd, on="nfl_id", how="left")
        tr_df = fold_df.iloc[tr_idx]
        va_df = fold_df.iloc[va_idx]
        y_tr = y[tr_idx]

        cnn_fold_data: tuple[np.ndarray, np.ndarray] | None = None
        if cnn_npz is not None and cnn_tag is not None:
            saved_tr = cnn_npz[f"{cnn_tag}_f{fold_i}_tr_idx"]
            saved_va = cnn_npz[f"{cnn_tag}_f{fold_i}_va_idx"]
            if not (np.array_equal(saved_tr, tr_idx) and np.array_equal(saved_va, va_idx)):
                raise ValueError(
                    f"Zero-leakage split parity violation for {cnn_tag} fold {fold_i}: "
                    "Colab CNN fold indices do not match models.py outer splits."
                )
            cnn_fold_data = (
                cnn_npz[f"{cnn_tag}_f{fold_i}_inner_oof"],
                cnn_npz[f"{cnn_tag}_f{fold_i}_va_feats"],
            )

        fold_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        for model_key, blocks in model_specs.items():
            oof_preds[model_key][va_idx] = _fit_predict_stacked_modalities(
                tr_df,
                va_df,
                blocks,
                y_tr,
                seed=seed + fold_i * 13,
                block_cache=fold_cache,
                cnn_fold_data=cnn_fold_data,
            )

    base_oof = oof_preds.get(baseline_key)
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model_key, pred in oof_preds.items():
        metrics_by_model[model_key] = bootstrap_metrics(
            y,
            pred,
            baseline_pred=None if model_key == baseline_key else base_oof,
            n_boot=n_boot,
            seed=seed,
        )
    return metrics_by_model, oof_preds, sub_outcomes


def compute_transfer_matrix(master_df: pd.DataFrame) -> dict[str, Any]:
    """Compute Combine-to-NFL Kinematic Transfer Matrix (position-adjusted Pearson r)."""
    features = [
        ("opt_power_index", "Optical Skill Power Index (z)", "optical"),
        ("opt_mass_power", "Optical Mass-Scaled Power (m·v·a)", "optical"),
        ("opt_impulse_momentum", "Optical Peak Impulse (m·a_max)", "optical"),
        ("opt_cbi_index", "Curvature Braking Index (CBI)", "optical"),
        ("opt_sk_max_s", "Optical Skill Peak Speed", "optical"),
        ("opt_f40_max_s", "Optical 40-Yd Peak Velocity", "optical"),
        ("opt_omega_z", "Optical Angular Cut Rate (ω)", "optical"),
        ("forty", "Official 40-Yd Stopwatch (inv)", "stopwatch"),
        ("ten_yd_split", "Official 10-Yd Split (inv)", "stopwatch"),
        ("vertical", "Vertical Jump (in)", "stopwatch"),
        ("broad_jump", "Broad Jump (in)", "stopwatch"),
    ]

    targets = [
        ("game_p99_s", "NFL Game Top Speed (p99)", "all", 1.0),
        ("game_p95_abs_a", "NFL Game Peak Burst (p95 |a|)", "all", 1.0),
        ("game_p95_power", "NFL Game Mech. Power", "all", 1.0),
        ("pp_get_off_mean", "DL First-Step Get-Off (faster)", "DL", -1.0),
        ("pp_yac_oe_mean", "WR/TE YAC Over Expected", "WRTE", 1.0),
        ("pp_sep_mean", "WR/TE Separation @ Throw", "WRTE", 1.0),
        ("snaps_per_season", "NFL Snaps / Season (All)", "all", 1.0),
        ("snaps_per_season", "OL Snaps / Season", "OL", 1.0),
    ]

    def _pos_z(s: pd.Series, pos: pd.Series) -> pd.Series:
        return s.groupby(pos).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    rows_out = []
    for f_col, f_label, f_kind in features:
        f_sign = -1.0 if f_col in ("forty", "ten_yd_split") else 1.0
        cells = []
        for t_col, t_label, scope, t_sign in targets:
            if scope == "all":
                sub = master_df
            elif scope == "WRTE":
                sub = master_df[master_df["combine_position"].isin(["WR", "TE"])]
            else:
                sub = master_df[master_df["combine_position"] == scope]

            m = sub[f_col].notna() & sub[t_col].notna()
            n_obs = int(m.sum())
            if n_obs < 20:
                cells.append({"target": t_label, "r": 0.0, "p": 1.0, "n": n_obs})
                continue
            x_val = sub.loc[m, f_col] * f_sign
            y_val = sub.loc[m, t_col] * t_sign
            if scope in ("all", "WRTE"):
                x_val = _pos_z(x_val, sub.loc[m, "combine_position"])
                y_val = _pos_z(y_val, sub.loc[m, "combine_position"])
            r_val, p_val = stats.pearsonr(x_val, y_val)
            cells.append({"target": t_label, "r": float(r_val), "p": float(p_val), "n": n_obs})
        rows_out.append({"feature": f_col, "label": f_label, "kind": f_kind, "cells": cells})

    return {"targets": [t[1] for t in targets], "rows": rows_out}


def compute_positional_archetypes(master_df: pd.DataFrame) -> dict[str, Any]:
    """Compute real position-specific biomechanical profiles across 6 optical axes."""
    axes = [
        ("opt_speed_index", "Skill Top Speed"),
        ("opt_burst_index", "Explosive Burst & Jerk"),
        ("opt_power_index", "Mechanical Power"),
        ("opt_cbi_index", "Curvature Braking (CBI)"),
        ("opt_omega_z", "Angular Cut Velocity"),
        ("opt_reaccel_z", "Out-of-Break Re-Accel"),
    ]
    profiles = {}
    for pos in COMBINE_POSITIONS:
        sub = master_df[master_df["combine_position"] == pos].copy()
        thresh = float(sub["snaps_per_season"].quantile(0.65))
        starters = sub[sub["snaps_per_season"] >= thresh]
        depth = sub[sub["snaps_per_season"] < thresh]

        axis_stats = []
        for col, label in axes:
            m_all = float(sub[col].mean())
            m_st = float(starters[col].mean())
            m_dp = float(depth[col].mean())
            axis_stats.append({
                "metric": col,
                "label": label,
                "pos_mean_z": m_all,
                "starter_mean_z": m_st,
                "depth_mean_z": m_dp,
                "starter_lift": m_st - m_dp,
            })

        best_r, best_col, best_p = 0.0, "", 1.0
        for col in [
            "opt_mass_power",
            "opt_impulse_momentum",
            "opt_power_index",
            "opt_cbi_index",
            "opt_omega_z",
            "opt_f40_max_a",
            "opt_f40_max_s",
        ]:
            m = sub[col].notna() & sub["snaps_per_season"].notna()
            if m.sum() >= 25:
                r, p = stats.pearsonr(sub.loc[m, col], sub.loc[m, "snaps_per_season"])
                if abs(r) > abs(best_r):
                    best_r, best_col, best_p = float(r), col, float(p)

        profiles[pos] = {
            "n_players": int(len(sub)),
            "starter_snap_threshold": thresh,
            "mean_mass_lbs": float(sub["combine_weight"].mean()),
            "mean_sk_max_s": float(sub["opt_sk_max_s"].mean()),
            "mean_sk_max_a": float(sub["opt_sk_max_a"].mean()),
            "mean_game_p99_s": float(sub["game_p99_s"].mean()),
            "top_optical_driver": {"feature": best_col, "r": best_r, "p": best_p},
            "axes": axis_stats,
        }
    return profiles


def run_full_modeling_suite(
    outcomes_df: pd.DataFrame,
    rep_df: pl.DataFrame,
    combine_results_pl: pl.DataFrame,
    n_boot: int = 2000,
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run all zero-leakage v1 + v2 Colab GPU 1D-CNN Deep Fusion benchmarks."""
    cnn_npz, cnn_meta = load_colab_cnn_artifacts(artifacts_dir)
    has_cnn = cnn_npz is not None

    full_kin_pd = aggregate_player_kinematics(rep_df, combine_results_pl, fit_nfl_ids=None).to_pandas()
    master_df = outcomes_df.merge(full_kin_pd, on="nfl_id", how="left")

    # Benchmark 1: On-Field NFL Game Kinematic Power Translation (22.65M frames, n=497)
    b1_specs: dict[str, list[list[str]]] = {
        "stopwatch_anthro": [STOPWATCH_COLS + ANTHRO_COLS],
        "ngs_scores": [NGS_COLS],
        "optical_only": [OPTICAL_CORE_COLS + ANTHRO_COLS],
        "stopwatch_plus_optical": [STOPWATCH_COLS + ANTHRO_COLS, OPTICAL_CORE_COLS + ANTHRO_COLS],
    }
    if has_cnn:
        b1_specs["cnn_1d_attention"] = [[CNN_RAW_BLOCK_KEY]]
        b1_specs["v2_deep_fusion"] = [
            OPTICAL_CORE_COLS + ANTHRO_COLS,
            [CNN_RAW_BLOCK_KEY],
        ]
        b1_specs["stopwatch_optical_cnn"] = [
            STOPWATCH_COLS + ANTHRO_COLS,
            OPTICAL_CORE_COLS + ANTHRO_COLS,
            [CNN_RAW_BLOCK_KEY],
        ]

    b1_strat_m, b1_strat_oof, b1_sub = run_zero_leakage_cv(
        outcomes_df, rep_df, combine_results_pl, "game_kinematic_index",
        b1_specs, cv_mode="stratified", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=cnn_npz, cnn_tag="b1_strat" if has_cnn else None,
    )
    b1_temp_m, b1_temp_oof, _ = run_zero_leakage_cv(
        outcomes_df, rep_df, combine_results_pl, "game_kinematic_index",
        b1_specs, cv_mode="temporal", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=cnn_npz, cnn_tag="b1_temp" if has_cnn else None,
    )

    y_b1 = b1_sub["game_kinematic_index"].to_numpy()
    b1_v2_vs_v1_lifts: dict[str, Any] = {}
    if has_cnn:
        b1_v2_vs_v1_lifts = {
            "stratified_v2_deep_vs_v1_stopwatch_optical": bootstrap_metrics(
                y_b1, b1_strat_oof["v2_deep_fusion"],
                baseline_pred=b1_strat_oof["stopwatch_plus_optical"], n_boot=n_boot,
            ),
            "stratified_v2_deep_vs_v1_optical_only": bootstrap_metrics(
                y_b1, b1_strat_oof["v2_deep_fusion"],
                baseline_pred=b1_strat_oof["optical_only"], n_boot=n_boot,
            ),
            "temporal_v2_deep_vs_v1_stopwatch_optical": bootstrap_metrics(
                y_b1, b1_temp_oof["v2_deep_fusion"],
                baseline_pred=b1_temp_oof["stopwatch_plus_optical"], n_boot=n_boot,
            ),
            "temporal_v2_deep_vs_v1_optical_only": bootstrap_metrics(
                y_b1, b1_temp_oof["v2_deep_fusion"],
                baseline_pred=b1_temp_oof["optical_only"], n_boot=n_boot,
            ),
        }

    # 40-Yard Dash Opt-Out Subset Evaluation
    m_optout = (b1_sub["opted_out_40"] == 1).to_numpy()
    optout_eval: dict[str, Any] = {
        "n_opt_out_players": int(m_optout.sum()),
        "stopwatch_anthro": bootstrap_metrics(
            y_b1[m_optout], b1_strat_oof["stopwatch_anthro"][m_optout], n_boot=n_boot
        ),
        "optical_only": bootstrap_metrics(
            y_b1[m_optout],
            b1_strat_oof["optical_only"][m_optout],
            baseline_pred=b1_strat_oof["stopwatch_anthro"][m_optout],
            n_boot=n_boot,
        ),
        "stopwatch_plus_optical": bootstrap_metrics(
            y_b1[m_optout],
            b1_strat_oof["stopwatch_plus_optical"][m_optout],
            baseline_pred=b1_strat_oof["stopwatch_anthro"][m_optout],
            n_boot=n_boot,
        ),
    }
    if has_cnn:
        optout_eval["cnn_1d_attention"] = bootstrap_metrics(
            y_b1[m_optout],
            b1_strat_oof["cnn_1d_attention"][m_optout],
            baseline_pred=b1_strat_oof["stopwatch_anthro"][m_optout],
            n_boot=n_boot,
        )
        optout_eval["v2_deep_fusion"] = bootstrap_metrics(
            y_b1[m_optout],
            b1_strat_oof["v2_deep_fusion"][m_optout],
            baseline_pred=b1_strat_oof["stopwatch_anthro"][m_optout],
            n_boot=n_boot,
        )

    # Benchmark 2: Pre-Draft Physical & Scouting Evaluation -> NFL Career Value (n=510)
    b2_specs: dict[str, list[list[str]]] = {
        "stopwatch_anthro": [STOPWATCH_COLS + ANTHRO_COLS],
        "optical_only": [OPTICAL_CORE_COLS + ANTHRO_COLS],
        "stopwatch_plus_optical": [STOPWATCH_COLS + ANTHRO_COLS, OPTICAL_CORE_COLS + ANTHRO_COLS],
        "ngs_only": [NGS_COLS],
        "ngs_plus_optical": [NGS_COLS, OPTICAL_CORE_COLS + ANTHRO_COLS],
        "draft_stopwatch": [DRAFT_COLS + STOPWATCH_COLS + ANTHRO_COLS],
        "draft_stopwatch_optical": [
            DRAFT_COLS + STOPWATCH_COLS + ANTHRO_COLS,
            OPTICAL_CORE_COLS + ANTHRO_COLS,
        ],
    }
    if has_cnn:
        b2_specs["cnn_1d_attention"] = [[CNN_RAW_BLOCK_KEY]]
        b2_specs["v2_deep_fusion"] = [
            STOPWATCH_COLS + ANTHRO_COLS,
            OPTICAL_CORE_COLS + ANTHRO_COLS,
            [CNN_MULTI_BLOCK_KEY],
        ]
        b2_specs["ngs_plus_v2_deep"] = [
            NGS_COLS,
            OPTICAL_CORE_COLS + ANTHRO_COLS,
            [CNN_MULTI_BLOCK_KEY],
        ]
        b2_specs["draft_plus_v2_deep"] = [
            DRAFT_COLS + STOPWATCH_COLS + ANTHRO_COLS,
            OPTICAL_CORE_COLS + ANTHRO_COLS,
            [CNN_MULTI_BLOCK_KEY],
        ]

    b2_strat_m, b2_strat_oof, b2_sub = run_zero_leakage_cv(
        outcomes_df, rep_df, combine_results_pl, "nfl_value_index",
        b2_specs, cv_mode="stratified", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=cnn_npz, cnn_tag="b2_strat" if has_cnn else None,
    )
    b2_temp_m, b2_temp_oof, _ = run_zero_leakage_cv(
        outcomes_df, rep_df, combine_results_pl, "nfl_value_index",
        b2_specs, cv_mode="temporal", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=cnn_npz, cnn_tag="b2_temp" if has_cnn else None,
    )

    y_b2 = b2_sub["nfl_value_index"].to_numpy()
    b2_paired_lifts: dict[str, Any] = {
        "stratified_ngs_plus_optical_vs_ngs": bootstrap_metrics(
            y_b2, b2_strat_oof["ngs_plus_optical"], baseline_pred=b2_strat_oof["ngs_only"], n_boot=n_boot
        ),
        "temporal_ngs_plus_optical_vs_ngs": bootstrap_metrics(
            y_b2, b2_temp_oof["ngs_plus_optical"], baseline_pred=b2_temp_oof["ngs_only"], n_boot=n_boot
        ),
        "stratified_draft_optical_vs_draft_stopwatch": bootstrap_metrics(
            y_b2,
            b2_strat_oof["draft_stopwatch_optical"],
            baseline_pred=b2_strat_oof["draft_stopwatch"],
            n_boot=n_boot,
        ),
        "temporal_draft_optical_vs_draft_stopwatch": bootstrap_metrics(
            y_b2,
            b2_temp_oof["draft_stopwatch_optical"],
            baseline_pred=b2_temp_oof["draft_stopwatch"],
            n_boot=n_boot,
        ),
    }
    if has_cnn:
        b2_paired_lifts.update({
            "stratified_v2_deep_vs_v1_stopwatch_optical": bootstrap_metrics(
                y_b2, b2_strat_oof["v2_deep_fusion"],
                baseline_pred=b2_strat_oof["stopwatch_plus_optical"], n_boot=n_boot,
            ),
            "temporal_v2_deep_vs_v1_stopwatch_optical": bootstrap_metrics(
                y_b2, b2_temp_oof["v2_deep_fusion"],
                baseline_pred=b2_temp_oof["stopwatch_plus_optical"], n_boot=n_boot,
            ),
            "stratified_ngs_v2_deep_vs_ngs": bootstrap_metrics(
                y_b2, b2_strat_oof["ngs_plus_v2_deep"],
                baseline_pred=b2_strat_oof["ngs_only"], n_boot=n_boot,
            ),
            "temporal_ngs_v2_deep_vs_ngs": bootstrap_metrics(
                y_b2, b2_temp_oof["ngs_plus_v2_deep"],
                baseline_pred=b2_temp_oof["ngs_only"], n_boot=n_boot,
            ),
            "stratified_draft_v2_deep_vs_draft": bootstrap_metrics(
                y_b2, b2_strat_oof["draft_plus_v2_deep"],
                baseline_pred=b2_strat_oof["draft_stopwatch"], n_boot=n_boot,
            ),
        })

    # Benchmark 3: DL/EDGE Regular-Season First-Step Get-Off (n=115)
    dl_sub = outcomes_df[(outcomes_df["combine_position"] == "DL") & (outcomes_df["pp_get_off_n"] >= 15)].copy()
    dl_specs: dict[str, list[list[str]]] = {
        "stopwatch_anthro": [STOPWATCH_COLS + ANTHRO_COLS],
        "optical_only": [OPTICAL_CORE_COLS + ANTHRO_COLS],
        "stopwatch_plus_optical": [STOPWATCH_COLS + ANTHRO_COLS, OPTICAL_CORE_COLS + ANTHRO_COLS],
    }
    if has_cnn:
        dl_specs["cnn_1d_attention"] = [[CNN_RAW_BLOCK_KEY]]
        dl_specs["v2_deep_fusion"] = [
            STOPWATCH_COLS + ANTHRO_COLS,
            [CNN_RAW_BLOCK_KEY],
        ]
    b3_dl_strat, _, _ = run_zero_leakage_cv(
        dl_sub, rep_df, combine_results_pl, "pp_get_off_mean",
        dl_specs, cv_mode="stratified", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=cnn_npz, cnn_tag="b3_strat" if has_cnn else None,
    )

    # Attach OOF predictions onto master_df
    b1_pred_dict: dict[str, Any] = {
        "nfl_id": b1_sub["nfl_id"],
        "oof_game_kin_stopwatch": b1_strat_oof["stopwatch_anthro"],
        "oof_game_kin_optical": b1_strat_oof["stopwatch_plus_optical"],
        "oof_game_kin_optical_only": b1_strat_oof["optical_only"],
    }
    if has_cnn:
        b1_pred_dict["oof_game_kin_cnn"] = b1_strat_oof["cnn_1d_attention"]
        b1_pred_dict["oof_game_kin_v2_fusion"] = b1_strat_oof["v2_deep_fusion"]

    b2_pred_dict: dict[str, Any] = {
        "nfl_id": b2_sub["nfl_id"],
        "oof_nfl_val_stopwatch": b2_strat_oof["stopwatch_anthro"],
        "oof_nfl_val_optical": b2_strat_oof["stopwatch_plus_optical"],
        "oof_nfl_val_optical_only": b2_strat_oof["optical_only"],
        "oof_nfl_val_ngs_optical": b2_strat_oof["ngs_plus_optical"],
        "oof_nfl_val_draft_optical": b2_strat_oof["draft_stopwatch_optical"],
        "oof_nfl_val_temporal_optical": b2_temp_oof["stopwatch_plus_optical"],
    }
    if has_cnn:
        b2_pred_dict["oof_nfl_val_cnn"] = b2_strat_oof["cnn_1d_attention"]
        b2_pred_dict["oof_nfl_val_v2_fusion"] = b2_strat_oof["v2_deep_fusion"]
        b2_pred_dict["oof_nfl_val_ngs_v2"] = b2_strat_oof["ngs_plus_v2_deep"]

    master_df = master_df.merge(pd.DataFrame(b1_pred_dict), on="nfl_id", how="left").merge(
        pd.DataFrame(b2_pred_dict), on="nfl_id", how="left"
    )
    primary_oof_col = "oof_nfl_val_v2_fusion" if has_cnn else "oof_nfl_val_optical"
    master_df["optical_alpha_signal"] = master_df[primary_oof_col] - master_df["oof_nfl_val_stopwatch"]

    def _serialize_player(row: pd.Series) -> dict[str, Any]:
        return {
            "nfl_id": int(row["nfl_id"]),
            "display_name": str(row["display_name"]),
            "draft_year": int(row["draft_year"]),
            "combine_position": str(row["combine_position"]),
            "nfl_position": str(row["nfl_position"]),
            "college_name": str(row["college_name"]),
            "draft_round": None if pd.isna(row["draft_round"]) else int(row["draft_round"]),
            "draft_overall_pick": None if pd.isna(row["draft_overall_pick"]) else int(row["draft_overall_pick"]),
            "forty": None if pd.isna(row["forty"]) else round(float(row["forty"]), 2),
            "ngs_athleticism_score": None if pd.isna(row["ngs_athleticism_score"]) else int(row["ngs_athleticism_score"]),
            "snaps_per_season": round(float(row["snaps_per_season"]), 1),
            "starts_per_season": round(float(row["starts_per_season"]), 1),
            "pro_bowl": int(row["pro_bowl_original_ballot"]),
            "all_pro_1st": int(row["ap_all_pro_1st_team"]),
            "all_pro_2nd": int(row["ap_all_pro_2nd_team"]),
            "nfl_value_index": round(float(row["nfl_value_index"]), 3),
            "draft_capital_alpha": round(float(row["draft_capital_alpha"]), 3),
            "oof_nfl_val_optical": round(float(row[primary_oof_col]), 3),
            "oof_game_kin_optical": None if pd.isna(row["oof_game_kin_optical"]) else round(float(row["oof_game_kin_optical"]), 3),
            "oof_game_kin_v2_fusion": None if (not has_cnn or pd.isna(row.get("oof_game_kin_v2_fusion"))) else round(float(row["oof_game_kin_v2_fusion"]), 3),
            "game_p99_s": None if pd.isna(row["game_p99_s"]) else round(float(row["game_p99_s"]), 2),
            "opt_power_index": round(float(row["opt_power_index"]), 3),
            "opt_cbi_index": round(float(row["opt_cbi_index"]), 3),
            "opt_sk_max_s": round(float(row["opt_sk_max_s"]), 2),
            "opt_sk_max_a": round(float(row["opt_sk_max_a"]), 2),
            "opt_mass_power": round(float(row["opt_mass_power"]), 1),
        }

    gems_df = (
        master_df[
            (master_df["draft_round"].isna() | (master_df["draft_round"] >= 4))
            & (master_df["draft_capital_alpha"] > 0.8)
            & (master_df[primary_oof_col] > 0.0)
        ]
        .sort_values("draft_capital_alpha", ascending=False)
        .head(10)
    )
    optout_gems_df = (
        master_df[(master_df["opted_out_40"] == 1) & (master_df["draft_capital_alpha"] > 0.5)]
        .sort_values("draft_capital_alpha", ascending=False)
        .head(8)
    )
    traps_df = (
        master_df[
            (master_df["draft_round"] <= 2)
            & (master_df["draft_capital_alpha"] < -0.6)
            & (master_df["snaps_per_season"] < 450.0)
            & (master_df["forty"].notna())
        ]
        .sort_values("draft_capital_alpha", ascending=True)
        .head(8)
    )
    all_alpha_df = master_df.sort_values("draft_capital_alpha", ascending=False).head(12)

    transfer_matrix = compute_transfer_matrix(master_df)
    archetypes = compute_positional_archetypes(master_df)

    q_labels = pd.qcut(master_df[primary_oof_col], q=5, labels=["Q1 (Bottom 20%)", "Q2", "Q3", "Q4", "Q5 (Top 20%)"])
    quintile_stats = []
    for ql in ["Q1 (Bottom 20%)", "Q2", "Q3", "Q4", "Q5 (Top 20%)"]:
        qs = master_df[q_labels == ql]
        quintile_stats.append({
            "quintile": ql,
            "n": int(len(qs)),
            "mean_snaps_per_season": round(float(qs["snaps_per_season"].mean()), 1),
            "mean_starts_per_season": round(float(qs["starts_per_season"].mean()), 2),
            "mean_nfl_value_z": round(float(qs["nfl_value_index"].mean()), 3),
            "starter_rate_500_snaps": round(float((qs["snaps_per_season"] >= 500).mean()), 3),
            "mean_game_p99_s": round(float(qs["game_p99_s"].mean()), 2),
        })

    results: dict[str, Any] = {
        "colab_gpu_temporal_cnn": cnn_meta if has_cnn else {"enabled": False},
        "benchmark_1_game_kinematic_translation": {
            "description": "Predict On-Field NFL Game Kinematic Power Index (22.65M game_tracking frames, n=497)",
            "stratified_5fold": b1_strat_m,
            "temporal_loyo": b1_temp_m,
            "forty_opt_out_subset": optout_eval,
            "v2_vs_v1_paired_lifts": b1_v2_vs_v1_lifts,
        },
        "benchmark_2_nfl_career_value": {
            "description": "Predict Exposure-Normalized NFL Career Value Index (510 rookies across 2023-2025 classes)",
            "stratified_5fold": b2_strat_m,
            "temporal_loyo": b2_temp_m,
            "paired_lifts": b2_paired_lifts,
        },
        "benchmark_3_dl_get_off": {
            "description": "Predict DL/EDGE Regular-Season First-Step Get-Off (player_play.csv, n=115)",
            "stratified_5fold": b3_dl_strat,
        },
        "optical_quintile_lift": quintile_stats,
        "transfer_matrix": transfer_matrix,
        "positional_archetypes": archetypes,
        "case_studies": {
            "day3_udfa_optical_gems": [_serialize_player(r) for _, r in gems_df.iterrows()],
            "forty_opt_out_stars": [_serialize_player(r) for _, r in optout_gems_df.iterrows()],
            "overdrafted_stopwatch_traps": [_serialize_player(r) for _, r in traps_df.iterrows()],
            "top_draft_capital_alpha": [_serialize_player(r) for _, r in all_alpha_df.iterrows()],
        },
    }
    return results, master_df
