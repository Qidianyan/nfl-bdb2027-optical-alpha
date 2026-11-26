#!/usr/bin/env python3
"""V4-iter1 Colab GPU Position-Specific Multi-Scale 1D-CNN + Dual-Head Temporal Attention
Trainer, Auto-Stop Dispatcher (COLAB_ACCOUNT=ultra-4), and Zero-Leakage Evaluator.

Modes:
  --mode train      : Runs on Colab CUDA GPU (NVIDIA L4/A100), training 4 position-specific
                      Multi-Scale (k=3,5,9) Dual-Head Temporal Attention encoders (WR_TE, DB,
                      DL_EDGE, OL_LB) under strictly zero-leakage Stratified 5-Fold & Temporal
                      LOYO GroupKFold CV + DL/EDGE First-Step Get-Off.
  --mode dispatch   : Orchestrates `COLAB_ACCOUNT=ultra-4 /Users/eligo/.local/bin/colab`
                      with guaranteed EXIT/INT/TERM GPU stop trap, runs `--mode train` on GPU,
                      downloads OOF artifacts, verifies 0.00/hr burn rate, then runs `--mode evaluate`.
  --mode evaluate   : Runs local zero-leakage V4 residual meta-fusion, 2,000x Bootstrap 95% CIs,
                      Partial R^2 vs Stopwatch & Draft+Stopwatch, BH-FDR q-values, and extracts
                      real 0.1s frame-by-frame dual-head saliency profiles to:
                        - materials/real_data_metrics_v4_gpu.json
                        - materials/v4_saliency_profiles.json
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package import works both locally (`code/nfl-bdb2027/scripts/`) and on Colab (`/content/`)
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from nfl_bdb2027.config import ARTIFACTS_DIR, COMBINE_POSITIONS, DEFAULT_DATA_DIR, DT_SEC, MATERIALS_DIR
    from nfl_bdb2027.positional_attn import (
        GROUP_TO_IDX,
        N_ENRICHED_CHANNELS,
        POS_ID_TO_GROUP,
        POSITION_GROUP_META,
        POSITION_GROUPS,
        PositionalResidualMetaFusion,
        benjamini_hochberg_fdr,
        bootstrap_ci_and_lift,
        build_enriched_10hz_channels,
        build_torch_positional_encoder,
        build_zero_leakage_group_splits,
        compute_partial_r2,
        extract_peak_saliency_window,
        fit_fold_channel_stats_v4,
        map_combine_position_to_group,
    )
except ImportError:
    # Colab flat /content/ layout fallback (positional_attn.py uploaded beside this script)
    from positional_attn import (  # type: ignore[no-redef]
        GROUP_TO_IDX,
        N_ENRICHED_CHANNELS,
        POS_ID_TO_GROUP,
        POSITION_GROUP_META,
        POSITION_GROUPS,
        build_enriched_10hz_channels,
        build_torch_positional_encoder,
        build_zero_leakage_group_splits,
        fit_fold_channel_stats_v4,
    )
    COMBINE_POSITIONS = ["WR", "DB", "DL", "OL", "TE"]
    DT_SEC = 0.1
    ARTIFACTS_DIR = Path("/content")
    MATERIALS_DIR = Path("/content")
    DEFAULT_DATA_DIR = Path("/content")


COLAB_BIN = "/Users/eligo/.local/bin/colab"
DEFAULT_TENSORS_NPZ = ARTIFACTS_DIR / "nfl_10hz_tensors.npz"
DEFAULT_V4_OOF_NPZ = ARTIFACTS_DIR / "v4_positional_attn_gpu_oof.npz"
DEFAULT_V4_GPU_META = ARTIFACTS_DIR / "v4_colab_gpu_metadata.json"
DEFAULT_METRICS_OUT = MATERIALS_DIR / "real_data_metrics_v4_gpu.json"
DEFAULT_SALIENCY_OUT = MATERIALS_DIR / "v4_saliency_profiles.json"

MAX_REPS = 18
SEQ_LEN = 96


# ---------------------------------------------------------------------------
# Within-Fold Anthropometric Standardization (Zero Leakage)
# ---------------------------------------------------------------------------

def _fit_transform_anthro(anthro: np.ndarray, pos_id: np.ndarray, tr_idx: np.ndarray) -> np.ndarray:
    n_p, d = anthro.shape
    out = np.zeros((n_p, d), dtype=np.float32)
    tr_a = anthro[tr_idx]
    tr_p = pos_id[tr_idx]
    for j in range(d):
        vals = tr_a[:, j]
        fin = vals[np.isfinite(vals)]
        gm = float(np.mean(fin)) if fin.size > 0 else 0.0
        gs = max(float(np.std(fin)), 1e-4) if fin.size > 1 else 1.0
        pm = np.full(len(COMBINE_POSITIONS), gm, dtype=np.float32)
        ps = np.full(len(COMBINE_POSITIONS), gs, dtype=np.float32)
        for p in range(len(COMBINE_POSITIONS)):
            pv = vals[tr_p == p]
            pv = pv[np.isfinite(pv)]
            if pv.size > 0:
                k = 10.0
                m_raw = float(np.mean(pv))
                s_raw = max(float(np.std(pv)), 1e-4) if pv.size > 1 else gs
                pm[p] = (pv.size * m_raw + k * gm) / (pv.size + k)
                ps[p] = max((pv.size * s_raw + k * gs) / (pv.size + k), 1e-4)
        raw = anthro[:, j]
        p_m = pm[pos_id]
        p_s = ps[pos_id]
        filled = np.where(np.isfinite(raw), raw, p_m)
        out[:, j] = np.clip((filled - p_m) / p_s, -4.5, 4.5)
    return out


# ---------------------------------------------------------------------------
# Colab CUDA GPU Training Loop (Two-Stage Shared Pretrain -> 4 Specialists)
# ---------------------------------------------------------------------------

def _huber_corr_loss(pred: "torch.Tensor", target: "torch.Tensor", corr_w: float = 0.25) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    fin = torch.isfinite(target)
    if fin.sum() < 2:
        return (pred * 0.0).sum()
    p = pred[fin]
    t = target[fin]
    huber = F.smooth_l1_loss(p, t, beta=0.75)
    if p.numel() >= 6:
        pc = p - p.mean()
        tc = t - t.mean()
        denom = torch.sqrt((pc.pow(2).sum() + 1e-6) * (tc.pow(2).sum() + 1e-6))
        corr = (pc * tc).sum() / denom
        return huber + corr_w * (1.0 - corr)
    return huber


def _augment_batch(x: "torch.Tensor", fm: "torch.Tensor", rm: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    # Mild sensor jitter + random single-rep dropout (mimics 40-yd/agility opt-out resilience)
    noise = torch.randn_like(x) * 0.045
    x_aug = (x + noise) * fm.unsqueeze(2)
    if rm.shape[1] > 4:
        keep_prob = 0.88
        drop_mask = (torch.rand_like(rm) < keep_prob).float()
        # Always preserve at least 2 valid reps
        rm_aug = rm * drop_mask
        too_few = rm_aug.sum(dim=1, keepdim=True) < 2.0
        rm_aug = torch.where(too_few, rm, rm_aug)
    else:
        rm_aug = rm
    return x_aug, rm_aug


def _train_model_epochs(
    model: "torch.nn.Module",
    x_tr: "torch.Tensor",
    fm_tr: "torch.Tensor",
    rm_tr: "torch.Tensor",
    did_tr: "torch.Tensor",
    gid_tr: "torch.Tensor",
    pid_tr: "torch.Tensor",
    an_tr: "torch.Tensor",
    y_tr: "torch.Tensor",
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    use_swa: bool = True,
) -> "torch.nn.Module":
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_tr = x_tr.shape[0]
    bs = min(batch_size, max(16, n_tr))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.05)

    swa_start = max(epochs - 10, int(epochs * 0.72))
    swa_state: dict[str, torch.Tensor] | None = None
    swa_n = 0

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=x_tr.device)
        for st in range(0, n_tr, bs):
            idx = perm[st : st + bs]
            if idx.numel() < 4:
                continue
            xb, rmb = _augment_batch(x_tr[idx], fm_tr[idx], rm_tr[idx])
            opt.zero_grad(set_to_none=True)
            preds_all, pred_skill, _ent, _pphys = model(
                xb, fm_tr[idx], rmb, did_tr[idx], gid_tr[idx], pid_tr[idx], an_tr[idx], return_attn=False
            )
            yb = y_tr[idx]
            loss = _huber_corr_loss(preds_all[:, 0], yb[:, 0], corr_w=0.30)
            if yb.shape[1] > 1:
                aux_loss = sum(
                    _huber_corr_loss(preds_all[:, j], yb[:, j], corr_w=0.15)
                    for j in range(1, yb.shape[1])
                ) / float(yb.shape[1] - 1)
                loss = loss + 0.35 * aux_loss
            loss = loss + 0.28 * _huber_corr_loss(pred_skill.squeeze(-1), yb[:, 0], corr_w=0.22)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if use_swa and ep >= swa_start:
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        swa_state[k] += v.detach()
            swa_n += 1

    if use_swa and swa_state is not None and swa_n > 0:
        for k in swa_state:
            if swa_state[k].is_floating_point():
                swa_state[k] /= float(swa_n)
        model.load_state_dict(swa_state)
    return model


def _eval_model(
    model: "torch.nn.Module",
    x_ev: "torch.Tensor",
    fm_ev: "torch.Tensor",
    rm_ev: "torch.Tensor",
    did_ev: "torch.Tensor",
    gid_ev: "torch.Tensor",
    pid_ev: "torch.Tensor",
    an_ev: "torch.Tensor",
    return_attn: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    import torch

    model.eval()
    with torch.no_grad():
        out = model(x_ev, fm_ev, rm_ev, did_ev, gid_ev, pid_ev, an_ev, return_attn=return_attn)
        preds_all, pred_skill, ent, p_phys = out[0], out[1], out[2], out[3]
        # Feature pack: [primary, aux_1..aux_k, pred_skill, ent, attn_phys(4)] -> n_targets + 6
        feats = torch.cat([preds_all, pred_skill, ent, p_phys], dim=-1).cpu().numpy().astype(np.float32)
        if return_attn:
            alpha_t = out[4].cpu().numpy().astype(np.float32)
            t_heads = out[5].cpu().numpy().astype(np.float32)
            m_w = out[6].cpu().numpy().astype(np.float32)
            return feats, alpha_t, t_heads, m_w
        return feats, None, None, None


def _run_positional_nested_cv_gpu(
    device: "torch.device",
    tag: str,
    subset_mask: np.ndarray,
    seq_x_all: np.ndarray,
    fmask_all: np.ndarray,
    rmask_all: np.ndarray,
    rep_len_all: np.ndarray,
    drill_id_all: np.ndarray,
    group_id_all: np.ndarray,
    pos_id_all: np.ndarray,
    nfl_ids_all: np.ndarray,
    draft_year_all: np.ndarray,
    anthro_all: np.ndarray,
    targets_all: np.ndarray,
    n_drills: int,
    cv_mode: str,
    pretrain_epochs: int = 36,
    specialist_epochs: int = 38,
    inner_folds: int = 4,
    seeds: tuple[int, ...] = (42, 1024),
    capture_attn: bool = False,
    only_group: str | None = None,
) -> dict[str, Any]:
    """Strictly zero-leakage nested outer+inner CV training 4 dedicated position-specific
    Multi-Scale (k=3,5,9) Dual-Head Temporal Attention encoders on CUDA GPU."""
    import copy
    import torch
    from scipy import stats
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, StratifiedKFold

    sub_indices = np.where(subset_mask)[0]
    seq_x = seq_x_all[sub_indices]
    fmask = fmask_all[sub_indices]
    rmask = rmask_all[sub_indices]
    rep_len = rep_len_all[sub_indices]
    drill_id = drill_id_all[sub_indices]
    group_id = group_id_all[sub_indices]
    pos_id = pos_id_all[sub_indices]
    nfl_ids = nfl_ids_all[sub_indices]
    draft_year = draft_year_all[sub_indices]
    anthro = anthro_all[sub_indices]
    targets = targets_all[sub_indices]

    comb_pos = np.asarray([COMBINE_POSITIONS[int(p)] for p in pos_id])
    pos_groups = np.asarray([POS_ID_TO_GROUP[int(p)] for p in pos_id])

    n_sub = len(sub_indices)
    n_targets = targets.shape[1]
    out_dim = n_targets + 6  # [primary, aux..., skill_pred, entropy, attn_phys(4)]

    outer_splits = build_zero_leakage_group_splits(
        nfl_ids, draft_year, comb_pos, mode=cv_mode, n_splits=5, seed=42
    )

    oof_feats = np.zeros((n_sub, out_dim), dtype=np.float32)
    oof_alpha = np.zeros((n_sub, MAX_REPS, SEQ_LEN), dtype=np.float32) if capture_attn else None
    oof_heads = np.zeros((n_sub, MAX_REPS, 2, SEQ_LEN), dtype=np.float32) if capture_attn else None
    oof_mil = np.zeros((n_sub, MAX_REPS), dtype=np.float32) if capture_attn else None

    fold_inner_oof: list[np.ndarray] = []
    fold_va_feats: list[np.ndarray] = []
    fold_tr_indices: list[np.ndarray] = []
    fold_va_indices: list[np.ndarray] = []

    models_trained = 0
    trainable_params = 0
    t0 = time.time()

    active_groups = (only_group,) if only_group is not None else POSITION_GROUPS

    for fold_i, (tr_idx, va_idx, fold_name) in enumerate(outer_splits):
        # 1. Fit 14-channel physics normalization & anthro scaling STRICTLY on outer tr_idx!
        ch_stats = fit_fold_channel_stats_v4(seq_x, fmask, rmask, drill_id, n_drills, tr_idx)
        x_norm = build_enriched_10hz_channels(seq_x, fmask, rep_len, drill_id, ch_stats)
        an_norm = _fit_transform_anthro(anthro, pos_id, tr_idx)

        # Strictly standardize targets per outer fold on tr_idx only (scale-invariant Huber+Corr for both B1 and B3!)
        y_tr_np = targets[tr_idx].copy()
        y_mu = np.nanmean(y_tr_np, axis=0, keepdims=True).astype(np.float32)
        y_sd = np.maximum(np.nanstd(y_tr_np, axis=0, keepdims=True), 1e-4).astype(np.float32)
        y_tr_scaled = (y_tr_np - y_mu) / y_sd

        # Transfer outer fold tensors to GPU once
        x_tr_t = torch.from_numpy(x_norm[tr_idx]).to(device)
        fm_tr_t = torch.from_numpy(fmask[tr_idx]).to(device)
        rm_tr_t = torch.from_numpy(rmask[tr_idx]).to(device)
        did_tr_t = torch.from_numpy(drill_id[tr_idx]).to(device)
        gid_tr_t = torch.from_numpy(group_id[tr_idx]).to(device)
        pid_tr_t = torch.from_numpy(pos_id[tr_idx]).to(device)
        an_tr_t = torch.from_numpy(an_norm[tr_idx]).to(device)
        y_tr_t = torch.from_numpy(y_tr_scaled).to(device)

        x_va_t = torch.from_numpy(x_norm[va_idx]).to(device)
        fm_va_t = torch.from_numpy(fmask[va_idx]).to(device)
        rm_va_t = torch.from_numpy(rmask[va_idx]).to(device)
        did_va_t = torch.from_numpy(drill_id[va_idx]).to(device)
        gid_va_t = torch.from_numpy(group_id[va_idx]).to(device)
        pid_va_t = torch.from_numpy(pos_id[va_idx]).to(device)
        an_va_t = torch.from_numpy(an_norm[va_idx]).to(device)

        tr_grp = pos_groups[tr_idx]
        va_grp = pos_groups[va_idx]

        # Inner CV strictly inside outer tr_idx
        tr_pos_ids = pos_id[tr_idx]
        uniq_c = np.bincount(tr_pos_ids)
        min_cls = int(uniq_c[uniq_c > 0].min())
        n_in = min(inner_folds, max(2, min_cls))
        if len(np.unique(tr_pos_ids)) > 1:
            in_splits = list(
                StratifiedKFold(n_splits=n_in, shuffle=True, random_state=42 + fold_i * 19).split(tr_idx, tr_pos_ids)
            )
        else:
            in_splits = list(KFold(n_splits=n_in, shuffle=True, random_state=42 + fold_i * 19).split(tr_idx))

        inner_oof = np.zeros((len(tr_idx), out_dim), dtype=np.float32)
        va_ens = np.zeros((len(va_idx), out_dim), dtype=np.float32)
        va_cnt = np.zeros((len(va_idx),), dtype=np.float32)

        va_a_acc = np.zeros((len(va_idx), MAX_REPS, SEQ_LEN), dtype=np.float32) if capture_attn else None
        va_h_acc = np.zeros((len(va_idx), MAX_REPS, 2, SEQ_LEN), dtype=np.float32) if capture_attn else None
        va_m_acc = np.zeros((len(va_idx), MAX_REPS), dtype=np.float32) if capture_attn else None

        for in_i, (itr, iva) in enumerate(in_splits):
            itr_t = torch.from_numpy(itr).to(device)

            # Stage A: Shared cross-position biomechanical pretraining strictly on itr (zero leakage!)
            base_model = build_torch_positional_encoder(
                in_channels=N_ENRICHED_CHANNELS,
                n_drills=n_drills,
                n_targets=n_targets,
                d_branch=16,
                d_rep=48,
                d_player=36,
                n_heads=2,
                dropout=0.28,
            ).to(device)
            if trainable_params == 0:
                trainable_params = int(sum(p.numel() for p in base_model.parameters() if p.requires_grad))

            base_seed = 700 + fold_i * 97 + in_i * 17
            if only_group is None:
                base_model = _train_model_epochs(
                    base_model,
                    x_tr_t[itr_t],
                    fm_tr_t[itr_t],
                    rm_tr_t[itr_t],
                    did_tr_t[itr_t],
                    gid_tr_t[itr_t],
                    pid_tr_t[itr_t],
                    an_tr_t[itr_t],
                    y_tr_t[itr_t],
                    epochs=pretrain_epochs,
                    lr=2.2e-3,
                    weight_decay=1.5e-2,
                    batch_size=64,
                    seed=base_seed,
                    use_swa=False,
                )
                models_trained += 1
            base_state = copy.deepcopy(base_model.state_dict())

            # Stage B: Fine-tune 4 dedicated Position-Specific Specialists strictly on itr & group == g!
            for grp in active_groups:
                g_itr = itr[tr_grp[itr] == grp]
                g_iva = iva[tr_grp[iva] == grp]
                g_va = np.where(va_grp == grp)[0]
                if len(g_itr) < 8:
                    continue

                g_itr_t = torch.from_numpy(g_itr).to(device)
                g_iva_t = torch.from_numpy(g_iva).to(device)
                g_va_t = torch.from_numpy(g_va).to(device)

                # Concatenate g_iva and g_va for one fast GPU eval pass
                if len(g_iva) > 0 and len(g_va) > 0:
                    x_ev = torch.cat([x_tr_t[g_iva_t], x_va_t[g_va_t]], dim=0)
                    fm_ev = torch.cat([fm_tr_t[g_iva_t], fm_va_t[g_va_t]], dim=0)
                    rm_ev = torch.cat([rm_tr_t[g_iva_t], rm_va_t[g_va_t]], dim=0)
                    did_ev = torch.cat([did_tr_t[g_iva_t], did_va_t[g_va_t]], dim=0)
                    gid_ev = torch.cat([gid_tr_t[g_iva_t], gid_va_t[g_va_t]], dim=0)
                    pid_ev = torch.cat([pid_tr_t[g_iva_t], pid_va_t[g_va_t]], dim=0)
                    an_ev = torch.cat([an_tr_t[g_iva_t], an_va_t[g_va_t]], dim=0)
                elif len(g_iva) > 0:
                    x_ev, fm_ev, rm_ev = x_tr_t[g_iva_t], fm_tr_t[g_iva_t], rm_tr_t[g_iva_t]
                    did_ev, gid_ev, pid_ev, an_ev = did_tr_t[g_iva_t], gid_tr_t[g_iva_t], pid_tr_t[g_iva_t], an_tr_t[g_iva_t]
                elif len(g_va) > 0:
                    x_ev, fm_ev, rm_ev = x_va_t[g_va_t], fm_va_t[g_va_t], rm_va_t[g_va_t]
                    did_ev, gid_ev, pid_ev, an_ev = did_va_t[g_va_t], gid_va_t[g_va_t], pid_va_t[g_va_t], an_va_t[g_va_t]
                else:
                    continue

                iva_acc = np.zeros((len(g_iva), out_dim), dtype=np.float32)
                for sd in seeds:
                    spec_model = build_torch_positional_encoder(
                        in_channels=N_ENRICHED_CHANNELS,
                        n_drills=n_drills,
                        n_targets=n_targets,
                        d_branch=16,
                        d_rep=48,
                        d_player=36,
                        n_heads=2,
                        dropout=0.28,
                    ).to(device)
                    if only_group is None:
                        spec_model.load_state_dict(base_state)
                        spec_lr = 1.1e-3
                        spec_ep = specialist_epochs
                    else:
                        spec_lr = 2.0e-3
                        spec_ep = pretrain_epochs + specialist_epochs

                    run_seed = sd + fold_i * 131 + in_i * 23 + GROUP_TO_IDX[grp] * 7
                    spec_model = _train_model_epochs(
                        spec_model,
                        x_tr_t[g_itr_t],
                        fm_tr_t[g_itr_t],
                        rm_tr_t[g_itr_t],
                        did_tr_t[g_itr_t],
                        gid_tr_t[g_itr_t],
                        pid_tr_t[g_itr_t],
                        an_tr_t[g_itr_t],
                        y_tr_t[g_itr_t],
                        epochs=spec_ep,
                        lr=spec_lr,
                        weight_decay=2.0e-2,
                        batch_size=32,
                        seed=run_seed,
                        use_swa=True,
                    )
                    models_trained += 1

                    ev_feats, ev_a, ev_h, ev_m = _eval_model(
                        spec_model, x_ev, fm_ev, rm_ev, did_ev, gid_ev, pid_ev, an_ev, return_attn=capture_attn
                    )
                    n_g_iva = len(g_iva)
                    if n_g_iva > 0:
                        iva_acc += ev_feats[:n_g_iva]
                    if len(g_va) > 0:
                        va_ens[g_va] += ev_feats[n_g_iva:]
                        va_cnt[g_va] += 1.0
                        if capture_attn and ev_a is not None and ev_h is not None and ev_m is not None:
                            va_a_acc[g_va] += ev_a[n_g_iva:]
                            va_h_acc[g_va] += ev_h[n_g_iva:]
                            va_m_acc[g_va] += ev_m[n_g_iva:]

                if len(g_iva) > 0:
                    inner_oof[g_iva] = iva_acc / float(len(seeds))

        denom_va = np.maximum(va_cnt[:, None], 1.0)
        va_ens /= denom_va
        # Inverse-scale primary & aux prediction heads back to original target units using tr_idx stats only
        va_ens[:, :n_targets] = va_ens[:, :n_targets] * y_sd + y_mu
        va_ens[:, n_targets] = va_ens[:, n_targets] * y_sd[0, 0] + y_mu[0, 0]
        inner_oof[:, :n_targets] = inner_oof[:, :n_targets] * y_sd + y_mu
        inner_oof[:, n_targets] = inner_oof[:, n_targets] * y_sd[0, 0] + y_mu[0, 0]
        oof_feats[va_idx] = va_ens

        if capture_attn and oof_alpha is not None and oof_heads is not None and oof_mil is not None:
            a_avg = va_a_acc / np.maximum(va_cnt[:, None, None], 1.0)
            h_avg = va_h_acc / np.maximum(va_cnt[:, None, None, None], 1.0)
            m_avg = va_m_acc / np.maximum(va_cnt[:, None], 1.0)
            # Re-normalize averaged frame attention strictly so sum(alpha_t) == 1.000000 on every valid rep
            fm_va_np = fmask[va_idx]
            a_avg = a_avg * (fm_va_np >= 0.5)
            a_sum = a_avg.sum(axis=-1, keepdims=True)
            a_avg = np.where(a_sum > 1e-12, a_avg / np.where(a_sum > 1e-12, a_sum, 1.0), 0.0)
            h_avg = h_avg * (fm_va_np[:, :, None, :] >= 0.5)
            h_sum = h_avg.sum(axis=-1, keepdims=True)
            h_avg = np.where(h_sum > 1e-12, h_avg / np.where(h_sum > 1e-12, h_sum, 1.0), 0.0)

            oof_alpha[va_idx] = a_avg
            oof_heads[va_idx] = h_avg
            oof_mil[va_idx] = m_avg

        fold_inner_oof.append(inner_oof)
        fold_va_feats.append(va_ens)
        fold_tr_indices.append(tr_idx)
        fold_va_indices.append(va_idx)
        fold_r2 = float(r2_score(targets[va_idx, 0], va_ens[:, 0]))
        print(f"    [{tag} | {fold_name:12s}] n_va={len(va_idx):3d} | Pos-Spec CNN R2={fold_r2:+.4f}", flush=True)

    y_true = targets[:, 0]
    y_pred = oof_feats[:, 0]
    r2 = float(r2_score(y_true, y_pred))
    rho = float(stats.spearmanr(y_true, y_pred)[0])
    elapsed = time.time() - t0
    pos_r2s = {}
    for grp in active_groups:
        gm = pos_groups == grp
        if int(gm.sum()) >= 10:
            pos_r2s[grp] = round(float(r2_score(y_true[gm], y_pred[gm])), 4)
    print(
        f"  [DONE {tag:18s} | {cv_mode:17s}] n={n_sub:3d} | Overall Pos-Spec R2={r2:+.4f}, rho={rho:+.4f} "
        f"| per-group={pos_r2s} ({elapsed:.1f}s, {models_trained} GPU models)",
        flush=True,
    )

    return {
        "sub_indices": sub_indices,
        "oof_feats": oof_feats,
        "oof_alpha": oof_alpha,
        "oof_heads": oof_heads,
        "oof_mil": oof_mil,
        "fold_inner_oof": fold_inner_oof,
        "fold_va_feats": fold_va_feats,
        "fold_tr_indices": fold_tr_indices,
        "fold_va_indices": fold_va_indices,
        "summary": {
            "n": int(n_sub),
            "r2": round(r2, 4),
            "spearman_rho": round(rho, 4),
            "per_group_r2": pos_r2s,
            "elapsed_sec": round(elapsed, 2),
            "models_trained": int(models_trained),
            "trainable_parameters_per_encoder": int(trainable_params),
        },
    }


def train_v4_on_gpu(
    tensors_npz: Path,
    out_npz: Path,
    out_meta_json: Path,
    pretrain_epochs: int = 36,
    specialist_epochs: int = 38,
) -> dict[str, Any]:
    """Run full V4 GPU training on Colab CUDA GPU and save OOF + 0.1s dual-head saliency bundle."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected! V4 GPU training must run on Colab NVIDIA L4/A100 GPU.")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    print(f"[V4-GPU] Device={gpu_name} ({vram_gb} GB VRAM) | PyTorch={torch.__version__}", flush=True)

    t_start = time.time()
    z = np.load(tensors_npz, allow_pickle=True)
    nfl_ids = z["nfl_ids"]
    pos_id = z["pos_id"]
    draft_year = z["draft_year"]
    anthro = z["anthro"]
    seq_x = z["seq_x"]
    fmask = z["frame_mask"]
    rmask = z["rep_mask"]
    rep_len = z["rep_len"]
    drill_id = z["rep_drill_id"]
    group_id = z["rep_group_id"]
    n_drills = len(z["canon_drills"])

    b1_targets = z["b1_targets"]
    b2_targets = z["b2_targets"]
    dl_get_off = z["dl_get_off"]
    dl_get_off_n = z["dl_get_off_n"]

    b1_mask = np.isfinite(b1_targets[:, 0])
    b2_mask = np.isfinite(b2_targets[:, 0])
    b3_mask = (pos_id == 2) & np.isfinite(dl_get_off) & (dl_get_off_n >= 10)
    b3_targets = np.stack([-dl_get_off, b1_targets[:, 0], b1_targets[:, 2]], axis=1).astype(np.float32)
    # Standardize B3 primary target safely inside each fold -> raw scale works directly with Huber+Corr

    runs: dict[str, dict[str, Any]] = {}
    # 1. Benchmark 1 (22.65M NFL Game Kinematic Power) - Stratified 5-Fold (capture exact OOF 0.1s saliency!)
    runs["v4_b1_strat"] = _run_positional_nested_cv_gpu(
        device, "v4_b1_strat", b1_mask, seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id,
        nfl_ids, draft_year, anthro, b1_targets, n_drills, cv_mode="stratified_player",
        pretrain_epochs=pretrain_epochs, specialist_epochs=specialist_epochs,
        inner_folds=4, seeds=(42, 1024), capture_attn=True,
    )
    # 2. Benchmark 1 - Temporal LOYO GroupKFold (2023/2024/2025)
    runs["v4_b1_temp"] = _run_positional_nested_cv_gpu(
        device, "v4_b1_temp", b1_mask, seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id,
        nfl_ids, draft_year, anthro, b1_targets, n_drills, cv_mode="temporal_loyo",
        pretrain_epochs=pretrain_epochs, specialist_epochs=specialist_epochs,
        inner_folds=4, seeds=(42, 1024), capture_attn=False,
    )
    # 3. Benchmark 3 - DL/EDGE First-Step Get-Off Specialist (n=115)
    runs["v4_b3_dl_strat"] = _run_positional_nested_cv_gpu(
        device, "v4_b3_dl_strat", b3_mask, seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id,
        nfl_ids, draft_year, anthro, b3_targets, n_drills, cv_mode="stratified_player",
        pretrain_epochs=32, specialist_epochs=38,
        inner_folds=4, seeds=(42, 1024), capture_attn=False, only_group="DL_EDGE",
    )

    # Pack OOF features, inner OOFs, outer indices, and exact OOF 0.1s dual-head attention tensors
    save_dict: dict[str, Any] = {
        "nfl_ids": nfl_ids,
        "pos_id": pos_id,
        "draft_year": draft_year,
    }
    summaries: dict[str, Any] = {}
    total_models = 0
    params_per_enc = 0
    for tag, res in runs.items():
        summaries[tag] = res["summary"]
        total_models += int(res["summary"]["models_trained"])
        if params_per_enc == 0:
            params_per_enc = int(res["summary"]["trainable_parameters_per_encoder"])
        save_dict[f"{tag}_sub_indices"] = res["sub_indices"]
        save_dict[f"{tag}_oof_feats"] = res["oof_feats"]
        for fi, (ioof, vaf, tri, vai) in enumerate(
            zip(res["fold_inner_oof"], res["fold_va_feats"], res["fold_tr_indices"], res["fold_va_indices"])
        ):
            save_dict[f"{tag}_f{fi}_inner_oof"] = ioof
            save_dict[f"{tag}_f{fi}_va_feats"] = vaf
            save_dict[f"{tag}_f{fi}_tr_idx"] = tri
            save_dict[f"{tag}_f{fi}_va_idx"] = vai
        if res["oof_alpha"] is not None:
            save_dict[f"{tag}_oof_alpha"] = res["oof_alpha"]
            save_dict[f"{tag}_oof_heads"] = res["oof_heads"]
            save_dict[f"{tag}_oof_mil"] = res["oof_mil"]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **save_dict)

    total_sec = round(time.time() - t_start, 2)
    meta = {
        "version": "v4-positional-attn-gpu",
        "colab_account": os.environ.get("COLAB_ACCOUNT", "ultra-4"),
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": str(torch.__version__),
        "architecture": "4 Position-Specific Multi-Scale 1D-CNN (k=3,5,9) + SE-ResBlock + FiLM + Explicit Dual-Head Temporal Attention + MIL",
        "position_groups": list(POSITION_GROUPS),
        "conv_kernels": [3, 5, 9],
        "n_attn_heads": 2,
        "enriched_channels": N_ENRICHED_CHANNELS,
        "trainable_parameters_per_encoder": params_per_enc,
        "total_specialist_parameters": params_per_enc * len(POSITION_GROUPS),
        "total_gpu_models_trained": total_models,
        "total_gpu_train_sec": total_sec,
        "total_combine_player_frames": int(z["total_player_frames"]),
        "packed_reps": int(z["packed_reps"]),
        "standalone_gpu_runs": summaries,
    }
    out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[V4-GPU DONE] Saved {out_npz} ({out_npz.stat().st_size / 1e6:.2f} MB) in {total_sec:.1f}s", flush=True)
    return meta


# ---------------------------------------------------------------------------
# Guaranteed-Cleanup Colab GPU Dispatcher (COLAB_ACCOUNT=ultra-4)
# ---------------------------------------------------------------------------

def dispatch_v4_to_colab(
    account: str = "ultra-4",
    preferred_gpu: str = "L4",
    tensors_npz: Path = DEFAULT_TENSORS_NPZ,
    out_oof_npz: Path = DEFAULT_V4_OOF_NPZ,
    out_meta_json: Path = DEFAULT_V4_GPU_META,
) -> None:
    """Launch Colab GPU session on `account`, run V4 training, download artifacts,
    and ALWAYS terminate GPU session via atexit + SIGINT/SIGTERM trap."""
    env = os.environ.copy()
    env["COLAB_ACCOUNT"] = account
    session = f"nfl2027-v4-posattn-{int(time.time())}"
    stopped = {"done": False}

    def _stop_gpu(*_args: Any) -> None:
        if stopped["done"]:
            return
        stopped["done"] = True
        print(f"[COLAB-TRAP] Stopping {session} on COLAB_ACCOUNT={account} ...", flush=True)
        subprocess.run([COLAB_BIN, "stop", "-s", session], env=env, check=False)
        subprocess.run([COLAB_BIN, "sessions"], env=env, check=False)
        subprocess.run([COLAB_BIN, "usage"], env=env, check=False)

    atexit.register(_stop_gpu)
    signal.signal(signal.SIGINT, lambda s, f: (_stop_gpu(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda s, f: (_stop_gpu(), sys.exit(143)))

    # 1. Start GPU (fast fallback A100 -> L4 if A100 unavailable)
    if preferred_gpu.upper() == "A100":
        env_fast = env.copy()
        env_fast["COLAB_CLI_ASSIGN_RETRIES"] = "0"
        rc = subprocess.run([COLAB_BIN, "new", "-s", session, "--gpu", "A100"], env=env_fast, check=False).returncode
        if rc != 0:
            subprocess.run([COLAB_BIN, "new", "-s", session, "--gpu", "L4"], env=env, check=True)
    else:
        subprocess.run([COLAB_BIN, "new", "-s", session, "--gpu", preferred_gpu], env=env, check=True)

    # 2. Verify GPU hardware
    subprocess.run(
        [COLAB_BIN, "ssh", "-s", session, "--", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"],
        env=env,
        stdin=subprocess.DEVNULL,
        check=True,
    )

    # 3. Upload tensor bundle + positional_attn.py + train_v4_colab_gpu.py
    pos_attn_py = SRC_DIR / "nfl_bdb2027" / "positional_attn.py"
    for local_f, remote_f in [
        (tensors_npz, "/content/nfl_10hz_tensors.npz"),
        (pos_attn_py, "/content/positional_attn.py"),
        (Path(__file__).resolve(), "/content/train_v4_colab_gpu.py"),
    ]:
        subprocess.run([COLAB_BIN, "upload", "-s", session, str(local_f), remote_f], env=env, check=True)

    # 4. Run training with --timeout 0 (never idle-timeout during CUDA epochs)
    launcher = Path("/tmp/nfl2027_v4_colab_launcher.py")
    launcher.write_text(
        "\n".join(
            [
                "import os, subprocess, sys",
                f"os.environ['COLAB_ACCOUNT'] = '{account}'",
                "cmd = [sys.executable, '/content/train_v4_colab_gpu.py', '--mode', 'train',",
                "       '--tensors-npz', '/content/nfl_10hz_tensors.npz',",
                "       '--out-npz', '/content/v4_positional_attn_gpu_oof.npz',",
                "       '--out-meta', '/content/v4_colab_gpu_metadata.json']",
                "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)",
                "for line in p.stdout: print(line, end='', flush=True)",
                "rc = p.wait()",
                "if rc != 0: raise RuntimeError(f'V4 GPU training failed rc={rc}')",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run([COLAB_BIN, "exec", "-s", session, "--timeout", "0", "-f", str(launcher)], env=env, check=True)
    launcher.unlink(missing_ok=True)

    # 5. Download trained V4 OOF + 0.1s dual-head saliency bundle & metadata
    subprocess.run([COLAB_BIN, "download", "-s", session, "/content/v4_positional_attn_gpu_oof.npz", str(out_oof_npz)], env=env, check=True)
    subprocess.run([COLAB_BIN, "download", "-s", session, "/content/v4_colab_gpu_metadata.json", str(out_meta_json)], env=env, check=True)

    # 6. Stop GPU immediately & verify zero burn
    _stop_gpu()


# ---------------------------------------------------------------------------
# Local Strictly Zero-Leakage V4 Meta-Fusion, BH-FDR, Bootstrap CI & Saliency Export
# ---------------------------------------------------------------------------

def _fit_fold_tabular_inner_and_va(
    tr_df: Any,
    va_df: Any,
    cols: list[str],
    y_tr: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce strictly leak-free inner-CV OOF and outer-va prediction for one tabular modality."""
    from sklearn.model_selection import KFold, StratifiedKFold
    from nfl_bdb2027.models import _fit_predict_modality

    va_pred = _fit_predict_modality(tr_df, va_df, cols, y_tr, seed=seed)
    tr_pos = tr_df["combine_position"].to_numpy()
    _, counts = np.unique(tr_pos, return_counts=True)
    n_in = min(5, max(2, int(counts.min())))
    if len(np.unique(tr_pos)) > 1:
        in_splits = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=seed).split(tr_df, tr_pos)
    else:
        in_splits = KFold(n_splits=n_in, shuffle=True, random_state=seed).split(tr_df)

    inner_oof = np.zeros(len(tr_df), dtype=np.float64)
    for k_i, (itr, iva) in enumerate(in_splits):
        inner_oof[iva] = _fit_predict_modality(
            tr_df.iloc[itr], tr_df.iloc[iva], cols, y_tr[itr], seed=seed + 31 * (k_i + 1)
        )
    return inner_oof, va_pred


def _fit_neural_block_inner_and_va(
    inner_feats: np.ndarray,
    va_feats: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit regularized RidgeCV projection of neural semantic outputs strictly inside outer fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    X_in = np.nan_to_num(inner_feats.astype(np.float64), nan=0.0)
    X_va = np.nan_to_num(va_feats.astype(np.float64), nan=0.0)
    mu = X_in.mean(axis=0, keepdims=True)
    sd = np.maximum(X_in.std(axis=0, keepdims=True), 1e-4)
    X_in_s = np.clip((X_in - mu) / sd, -4.5, 4.5)
    X_va_s = np.clip((X_va - mu) / sd, -4.5, 4.5)

    alphas = np.logspace(-0.5, 3.0, 18)
    full_m = RidgeCV(alphas=alphas).fit(X_in_s, y_tr)
    va_pred = full_m.predict(X_va_s)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    in_oof = np.zeros(len(y_tr), dtype=np.float64)
    for itr, iva in kf.split(X_in_s):
        in_oof[iva] = RidgeCV(alphas=alphas).fit(X_in_s[itr], y_tr[itr]).predict(X_in_s[iva])
    return in_oof, va_pred


def _extract_saliency_profiles(
    tensors_npz: Path,
    v4_oof_npz: Path,
    master_df: Any,
    out_saliency_json: Path,
) -> dict[str, Any]:
    """Build real 0.1s frame-by-frame dual-head temporal attention saliency profiles
    for featured Day-3/Opt-Out Gem vs Early-Round Stopwatch Trap matchups across all 4 position groups."""
    z = np.load(tensors_npz, allow_pickle=True)
    v4 = np.load(v4_oof_npz, allow_pickle=True)

    sub_idx = v4["v4_b1_strat_sub_indices"]
    oof_alpha = v4["v4_b1_strat_oof_alpha"]  # (497, 18, 96)
    oof_heads = v4["v4_b1_strat_oof_heads"]  # (497, 18, 2, 96)
    oof_mil = v4["v4_b1_strat_oof_mil"]      # (497, 18)

    nfl_ids = z["nfl_ids"]
    id_to_sub = {int(nfl_ids[int(gi)]): si for si, gi in enumerate(sub_idx)}
    id_to_glob = {int(pid): gi for gi, pid in enumerate(nfl_ids)}
    drills = z["canon_drills"].astype(str)

    seq_x = z["seq_x"]
    fmask = z["frame_mask"]
    rmask = z["rep_mask"]
    rep_drill_id = z["rep_drill_id"]
    rep_len = z["rep_len"]

    df_idx = master_df.set_index("nfl_id")

    def _find_player_by_name(name: str) -> int | None:
        m = master_df[master_df["display_name"] == name]
        return int(m.iloc[0]["nfl_id"]) if len(m) > 0 else None

    def _extract_player_drill(pid: int, preferred_drills: list[str]) -> dict[str, Any]:
        gi = id_to_glob[pid]
        si = id_to_sub[pid]
        row = df_idx.loc[pid]
        n_reps = int(rmask[gi].sum())
        chosen_r = 0
        chosen_drill = drills[int(rep_drill_id[gi, 0])]
        found = False
        for pref in preferred_drills:
            for r in range(n_reps):
                if drills[int(rep_drill_id[gi, r])] == pref:
                    chosen_r = r
                    chosen_drill = pref
                    found = True
                    break
            if found:
                break
        if not found:
            # Fallback to highest MIL attention weight rep
            chosen_r = int(np.argmax(oof_mil[si, :n_reps]))
            chosen_drill = drills[int(rep_drill_id[gi, chosen_r])]

        nf = int(rep_len[gi, chosen_r])
        alpha = oof_alpha[si, chosen_r, :nf].astype(np.float64)
        # Strict normalization guarantee
        s_a = float(alpha.sum())
        if s_a > 1e-12:
            alpha = alpha / s_a
        h1 = oof_heads[si, chosen_r, 0, :nf].astype(np.float64)
        h2 = oof_heads[si, chosen_r, 1, :nf].astype(np.float64)
        if h1.sum() > 1e-12:
            h1 = h1 / h1.sum()
        if h2.sum() > 1e-12:
            h2 = h2 / h2.sum()

        s_arr = seq_x[gi, chosen_r, 0, :nf].astype(np.float64)
        atan_arr = seq_x[gi, chosen_r, 1, :nf].astype(np.float64)
        om_arr = seq_x[gi, chosen_r, 2, :nf].astype(np.float64)
        acent_arr = s_arr * om_arr
        win = extract_peak_saliency_window(alpha, np.ones(nf, dtype=np.float32), window_frames=3, dt_sec=DT_SEC)

        w_st, w_en = win["start_frame"], win["end_frame"] + 1
        win["mean_speed_yd_s"] = round(float(np.mean(s_arr[w_st:w_en])), 3)
        win["min_a_tan_yd_s2"] = round(float(np.min(atan_arr[w_st:w_en])), 3)
        win["max_a_cent_yd_s2"] = round(float(np.max(acent_arr[w_st:w_en])), 3)

        dr = None if np.isnan(row["draft_round"]) else int(row["draft_round"])
        pk = None if np.isnan(row["draft_overall_pick"]) else int(row["draft_overall_pick"])
        f40 = None if np.isnan(row["forty"]) else round(float(row["forty"]), 2)
        return {
            "nfl_id": int(pid),
            "display_name": str(row["display_name"]),
            "draft_year": int(row["draft_year"]),
            "combine_position": str(row["combine_position"]),
            "position_group": map_combine_position_to_group(str(row["combine_position"])),
            "draft_round": dr,
            "draft_overall_pick": pk,
            "forty_sec": f40,
            "draft_capital_alpha": round(float(row["draft_capital_alpha"]), 3),
            "snaps_per_season": round(float(row["snaps_per_season"]), 1),
            "drill_name": str(chosen_drill),
            "rep_mil_weight": round(float(oof_mil[si, chosen_r]), 4),
            "n_frames": int(nf),
            "attn_sum": round(float(alpha.sum()), 6),
            "peak_0_3s_window": win,
            "frames": [
                {
                    "frame_idx": int(t),
                    "t_sec": round(t * DT_SEC, 1),
                    "speed_yd_s": round(float(s_arr[t]), 3),
                    "a_tan_yd_s2": round(float(atan_arr[t]), 3),
                    "omega_rad_s": round(float(om_arr[t]), 3),
                    "a_cent_yd_s2": round(float(acent_arr[t]), 3),
                    "attn_weight": round(float(alpha[t]), 6),
                    "attn_head1_burst": round(float(h1[t]), 6),
                    "attn_head2_brake": round(float(h2[t]), 6),
                }
                for t in range(nf)
            ],
        }

    # Select best real OL Day-3 gem vs Early-Round OL trap automatically from master_df
    ol_sub = master_df[(master_df["combine_position"] == "OL") & (master_df["nfl_id"].isin(id_to_sub))].copy()
    ol_gems = ol_sub[(ol_sub["draft_round"].isna()) | (ol_sub["draft_round"] >= 4)].sort_values("draft_capital_alpha", ascending=False)
    ol_traps = ol_sub[ol_sub["draft_round"] <= 2].sort_values("draft_capital_alpha", ascending=True)
    ol_gem_id = int(ol_gems.iloc[0]["nfl_id"])
    ol_trap_id = int(ol_traps.iloc[0]["nfl_id"])

    matchup_specs = [
        (
            "WR_TE_gauntlet_nacua_vs_polk",
            "WR_TE",
            "40-Yd Opt-Out Day-3 Gem vs Round 2 WR: Decisive 0.3s Gauntlet Braking & Cut Attention",
            _find_player_by_name("Puka Nacua"),
            _find_player_by_name("Ja'Lynn Polk"),
            ["GAUNTLET_DRILL", "COMEBACK_ROUTE_RIGHT", "GO_ROUTE_RIGHT"],
        ),
        (
            "DB_break90_jones_vs_smith",
            "DB",
            "Rd 7 Cornerback Gem vs Faster-40 Rd 2 Trap: 90-Degree Hip-Flip & Braking Saliency",
            _find_player_by_name("Jaylon Jones"),
            _find_player_by_name("Cam Smith"),
            ["BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "FORTY_YARD_DASH"],
        ),
        (
            "DL_EDGE_passrush_tuipulotu_vs_vanness",
            "DL_EDGE",
            "40-Yd Opt-Out Rush Star vs Rd 1 #13 Stopwatch Freak: Pass-Rush First-Step Saliency",
            _find_player_by_name("Tuli Tuipulotu"),
            _find_player_by_name("Lukas Van Ness"),
            ["PASS_RUSH_DRILL", "BODY_CONTROL_DRILL", "SHORT_ZONE_BREAKS"],
        ),
        (
            "OL_LB_trench_gem_vs_trap",
            "OL_LB",
            "Day-3 Trench Starter vs Early-Round OL Underperformer: Lateral Mirror & Pull Stability",
            ol_gem_id,
            ol_trap_id,
            ["PASS_PRO-MIRROR_DRILL", "LINE_DRILL", "FORTY_YARD_DASH"],
        ),
    ]

    matchups = []
    for mid, grp, title, gem_id, trap_id, prefs in matchup_specs:
        if gem_id is None or trap_id is None:
            continue
        gem_prof = _extract_player_drill(gem_id, prefs)
        trap_prof = _extract_player_drill(trap_id, prefs)
        matchups.append(
            {
                "matchup_id": mid,
                "position_group": grp,
                "title": title,
                "gem": gem_prof,
                "trap": trap_prof,
                "scouting_takeaway": (
                    f"{gem_prof['display_name']} peaks at {gem_prof['peak_0_3s_window']['start_sec']:.1f}-"
                    f"{gem_prof['peak_0_3s_window']['end_sec']:.1f}s "
                    f"(attn mass {gem_prof['peak_0_3s_window']['attention_mass']*100:.1f}%, "
                    f"speed {gem_prof['peak_0_3s_window']['mean_speed_yd_s']:.2f} yd/s, "
                    f"cut {gem_prof['peak_0_3s_window']['max_a_cent_yd_s2']:.2f} yd/s^2) vs "
                    f"{trap_prof['display_name']} ({trap_prof['peak_0_3s_window']['start_sec']:.1f}-"
                    f"{trap_prof['peak_0_3s_window']['end_sec']:.1f}s, "
                    f"speed {trap_prof['peak_0_3s_window']['mean_speed_yd_s']:.2f} yd/s)."
                ),
            }
        )

    # Also compute cohort-level mean peak 0.3s saliency window per position group across all 497 tracked players
    cohort_summary: dict[str, Any] = {}
    for grp in POSITION_GROUPS:
        win_starts, win_masses, top_drills = [], [], []
        for si, gi in enumerate(sub_idx):
            g_str = POS_ID_TO_GROUP[int(z["pos_id"][int(gi)])]
            if g_str != grp:
                continue
            nr = int(rmask[int(gi)].sum())
            if nr == 0:
                continue
            best_r = int(np.argmax(oof_mil[si, :nr]))
            nf = int(rep_len[int(gi), best_r])
            win = extract_peak_saliency_window(oof_alpha[si, best_r, :nf], np.ones(nf, dtype=np.float32), 3, DT_SEC)
            win_starts.append(win["start_sec"])
            win_masses.append(win["attention_mass"])
            top_drills.append(str(drills[int(rep_drill_id[int(gi), best_r])]))
        # Most frequent MIL-attended drill in this position group
        drill_counts: dict[str, int] = {}
        for d in top_drills:
            drill_counts[d] = drill_counts.get(d, 0) + 1
        top_3 = sorted(drill_counts.items(), key=lambda kv: -kv[1])[:3]
        cohort_summary[grp] = {
            "n_players": len(win_starts),
            "mean_peak_window_start_sec": round(float(np.mean(win_starts)), 2) if win_starts else 0.0,
            "mean_peak_window_end_sec": round(float(np.mean(win_starts)) + 0.30, 2) if win_starts else 0.3,
            "mean_0_3s_attention_mass": round(float(np.mean(win_masses)), 4) if win_masses else 0.0,
            "top_mil_attended_drills": [{"drill": d, "count": int(c)} for d, c in top_3],
        }

    payload = {
        "version": "v4-positional-attn-saliency",
        "sampling_hz": 10.0,
        "dt_sec": DT_SEC,
        "window_sec": 0.3,
        "saliency_source": "Strictly out-of-fold Colab GPU V4 Position-Specific Dual-Head Temporal Attention weights (sum(alpha_t) == 1.0)",
        "cohort_position_saliency": cohort_summary,
        "matchups": matchups,
    }
    out_saliency_json.parent.mkdir(parents=True, exist_ok=True)
    out_saliency_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SALIENCY] Wrote {out_saliency_json} ({len(matchups)} matchups across 4 position groups)")
    return payload


def evaluate_v4_locally(
    data_dir: Path = DEFAULT_DATA_DIR,
    tensors_npz: Path = DEFAULT_TENSORS_NPZ,
    v4_oof_npz: Path = DEFAULT_V4_OOF_NPZ,
    v4_gpu_meta: Path = DEFAULT_V4_GPU_META,
    out_metrics_json: Path = DEFAULT_METRICS_OUT,
    out_saliency_json: Path = DEFAULT_SALIENCY_OUT,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Run strictly zero-leakage V4 evaluation & residual meta-fusion against V2 and Stopwatch baselines."""
    from nfl_bdb2027.data import DatasetPaths, load_combine_results
    from nfl_bdb2027.kinematics import aggregate_player_kinematics, extract_rep_level_kinematics
    from nfl_bdb2027.models import (
        ANTHRO_COLS,
        DRAFT_COLS,
        OPTICAL_CORE_COLS,
        STOPWATCH_COLS,
        load_colab_cnn_artifacts,
        run_zero_leakage_cv,
    )
    from nfl_bdb2027.outcomes import build_outcomes_table

    t0 = time.time()
    paths = DatasetPaths.from_dir(data_dir)
    rep_df = extract_rep_level_kinematics(paths)
    cr_pl = load_combine_results(paths)
    outcomes_df = build_outcomes_table(paths)
    full_kin = aggregate_player_kinematics(rep_df, cr_pl, fit_nfl_ids=None).to_pandas()
    master_df = outcomes_df.merge(full_kin, on="nfl_id", how="left")

    v2_npz, _v2_meta = load_colab_cnn_artifacts(ARTIFACTS_DIR)
    assert v2_npz is not None, "Missing V2 baseline colab_temporal_cnn_oof.npz"
    v4_npz = dict(np.load(v4_oof_npz, allow_pickle=True))
    gpu_meta = json.loads(v4_gpu_meta.read_text(encoding="utf-8"))

    # Exact V2 & Stopwatch baselines on identical zero-leakage splits
    v2_metrics_file = json.loads((MATERIALS_DIR / "real_data_metrics_v2.json").read_text(encoding="utf-8"))
    b1_specs = {
        "stopwatch_anthro": [STOPWATCH_COLS + ANTHRO_COLS],
        "draft_stopwatch": [DRAFT_COLS + STOPWATCH_COLS + ANTHRO_COLS],
        "optical_only": [OPTICAL_CORE_COLS + ANTHRO_COLS],
        "v2_deep_fusion": [OPTICAL_CORE_COLS + ANTHRO_COLS, ["__CNN_1D_ATTENTION_RAW__"]],
    }
    b1_s_m, b1_s_oof, b1_sub = run_zero_leakage_cv(
        outcomes_df, rep_df, cr_pl, "game_kinematic_index", b1_specs,
        cv_mode="stratified", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=v2_npz, cnn_tag="b1_strat",
    )
    b1_t_m, b1_t_oof, _ = run_zero_leakage_cv(
        outcomes_df, rep_df, cr_pl, "game_kinematic_index", b1_specs,
        cv_mode="temporal", baseline_key="stopwatch_anthro", n_boot=n_boot,
        cnn_npz=v2_npz, cnn_tag="b1_temp",
    )

    y_b1 = b1_sub["game_kinematic_index"].to_numpy(dtype=np.float64)
    b1_ids = b1_sub["nfl_id"].to_numpy()
    b1_yrs = b1_sub["draft_year"].to_numpy()
    b1_cpos = b1_sub["combine_position"].to_numpy()
    b1_groups = np.asarray([map_combine_position_to_group(p) for p in b1_cpos])

    def _run_v4_fusion_cv(cv_mode: str, v4_tag: str, v2_tag: str) -> tuple[np.ndarray, np.ndarray]:
        splits = build_zero_leakage_group_splits(b1_ids, b1_yrs, b1_cpos, mode=cv_mode, n_splits=5, seed=42)
        v4_fused_oof = np.zeros(len(b1_sub), dtype=np.float64)
        v4_standalone_oof = np.zeros(len(b1_sub), dtype=np.float64)
        fuser = PositionalResidualMetaFusion(l2_reg=0.06, residual_blend=0.35)

        for fi, (tr_idx, va_idx, _fname) in enumerate(splits):
            # Strict split parity verification against both V2 and V4 GPU artifacts
            assert np.array_equal(v4_npz[f"{v4_tag}_f{fi}_tr_idx"], tr_idx), f"V4 train split mismatch {v4_tag} f{fi}"
            assert np.array_equal(v4_npz[f"{v4_tag}_f{fi}_va_idx"], va_idx), f"V4 val split mismatch {v4_tag} f{fi}"
            assert np.array_equal(v2_npz[f"{v2_tag}_f{fi}_tr_idx"], tr_idx), f"V2 train split mismatch {v2_tag} f{fi}"
            assert np.array_equal(v2_npz[f"{v2_tag}_f{fi}_va_idx"], va_idx), f"V2 val split mismatch {v2_tag} f{fi}"

            fit_ids = b1_sub.iloc[tr_idx]["nfl_id"].tolist()
            kin_fold = aggregate_player_kinematics(rep_df, cr_pl, fit_nfl_ids=fit_ids).to_pandas()
            fold_df = b1_sub.merge(kin_fold, on="nfl_id", how="left")
            tr_df = fold_df.iloc[tr_idx]
            va_df = fold_df.iloc[va_idx]
            y_tr = y_b1[tr_idx]

            # Block 1: Optical tabular physics (strictly within-fold)
            opt_in, opt_va = _fit_fold_tabular_inner_and_va(
                tr_df, va_df, OPTICAL_CORE_COLS + ANTHRO_COLS, y_tr, seed=42 + fi * 13
            )
            # Block 2: V2 unified 1D-CNN semantic head (strictly within-fold inner OOF)
            v2_in, v2_va = _fit_neural_block_inner_and_va(
                v2_npz[f"{v2_tag}_f{fi}_inner_oof"], v2_npz[f"{v2_tag}_f{fi}_va_feats"], y_tr, seed=142 + fi * 13
            )
            # Block 3: V4 Position-Specific Multi-Scale Dual-Head Attention GPU specialist
            v4_tr_raw = v4_npz[f"{v4_tag}_f{fi}_inner_oof"]
            v4_va_raw = v4_npz[f"{v4_tag}_f{fi}_va_feats"]
            v4_in, v4_va = _fit_neural_block_inner_and_va(v4_tr_raw, v4_va_raw, y_tr, seed=242 + fi * 13)
            v4_standalone_oof[va_idx] = v4_va_raw[:, 0]

            # Block 4: Stopwatch + Anthro (complementary linear velocity signal)
            sw_in, sw_va = _fit_fold_tabular_inner_and_va(
                tr_df, va_df, STOPWATCH_COLS + ANTHRO_COLS, y_tr, seed=342 + fi * 13
            )

            tr_blocks = np.column_stack([opt_in, v2_in, v4_in, sw_in])
            va_blocks = np.column_stack([opt_va, v2_va, v4_va, sw_va])

            v4_fused_oof[va_idx] = fuser.fit_predict_fold(
                tr_blocks_inner=tr_blocks,
                va_blocks=va_blocks,
                tr_pos_extra=v4_tr_raw,
                va_pos_extra=v4_va_raw,
                tr_groups=b1_groups[tr_idx],
                va_groups=b1_groups[va_idx],
                y_tr=y_tr,
            )

        return v4_fused_oof, v4_standalone_oof

    v4_strat_oof, v4_strat_cnn = _run_v4_fusion_cv("stratified_player", "v4_b1_strat", "b1_strat")
    v4_temp_oof, v4_temp_cnn = _run_v4_fusion_cv("temporal_loyo", "v4_b1_temp", "b1_temp")

    # Benchmark 3: DL/EDGE First-Step Get-Off (n=115) fusion with V4 DL_EDGE specialist
    dl_sub = (
        outcomes_df[(outcomes_df["combine_position"] == "DL") & outcomes_df["pp_get_off_mean"].notna() & (outcomes_df["pp_get_off_n"] >= 10)]
        .sort_values("nfl_id")
        .reset_index(drop=True)
    )
    y_dl = -dl_sub["pp_get_off_mean"].to_numpy(dtype=np.float64)
    dl_splits = build_zero_leakage_group_splits(
        dl_sub["nfl_id"].to_numpy(), dl_sub["draft_year"].to_numpy(), dl_sub["combine_position"].to_numpy(),
        mode="stratified_player", n_splits=5, seed=42,
    )
    v4_dl_oof = np.zeros(len(dl_sub), dtype=np.float64)
    sw_dl_oof = np.zeros(len(dl_sub), dtype=np.float64)
    for fi, (tr_idx, va_idx, _) in enumerate(dl_splits):
        fit_ids = dl_sub.iloc[tr_idx]["nfl_id"].tolist()
        kin_fold = aggregate_player_kinematics(rep_df, cr_pl, fit_nfl_ids=fit_ids).to_pandas()
        fold_df = dl_sub.merge(kin_fold, on="nfl_id", how="left")
        tr_df, va_df = fold_df.iloc[tr_idx], fold_df.iloc[va_idx]
        y_tr = y_dl[tr_idx]
        sw_in, sw_va = _fit_fold_tabular_inner_and_va(tr_df, va_df, STOPWATCH_COLS + ANTHRO_COLS, y_tr, seed=42 + fi * 13)
        opt_in, opt_va = _fit_fold_tabular_inner_and_va(tr_df, va_df, OPTICAL_CORE_COLS + ANTHRO_COLS, y_tr, seed=142 + fi * 13)
        v2_dl_in, v2_dl_va = _fit_neural_block_inner_and_va(
            v2_npz[f"b3_strat_f{fi}_inner_oof"], v2_npz[f"b3_strat_f{fi}_va_feats"], y_tr, seed=242 + fi * 13
        )
        v4_dl_in, v4_dl_va = _fit_neural_block_inner_and_va(
            v4_npz[f"v4_b3_dl_strat_f{fi}_inner_oof"], v4_npz[f"v4_b3_dl_strat_f{fi}_va_feats"], y_tr, seed=342 + fi * 13
        )
        sw_dl_oof[va_idx] = sw_va
        fuser_dl = PositionalResidualMetaFusion(l2_reg=0.06, residual_blend=0.15)
        v4_dl_oof[va_idx] = fuser_dl.fit_predict_fold(
            np.column_stack([opt_in, v2_dl_in, v4_dl_in]),
            np.column_stack([opt_va, v2_dl_va, v4_dl_va]),
            v4_npz[f"v4_b3_dl_strat_f{fi}_inner_oof"],
            v4_npz[f"v4_b3_dl_strat_f{fi}_va_feats"],
            np.full(len(tr_idx), "DL_EDGE"),
            np.full(len(va_idx), "DL_EDGE"),
            y_tr,
        )

    # Extract 0.1s saliency profiles
    saliency_payload = _extract_saliency_profiles(tensors_npz, v4_oof_npz, master_df, out_saliency_json)

    # Overall statistical evaluation (2,000x Bootstrap CIs + Partial R^2 + BH-FDR!)
    sw_strat_oof = b1_s_oof["stopwatch_anthro"]
    draft_sw_strat_oof = b1_s_oof["draft_stopwatch"]
    v2_strat_oof = b1_s_oof["v2_deep_fusion"]
    sw_temp_oof = b1_t_oof["stopwatch_anthro"]
    v2_temp_oof = b1_t_oof["v2_deep_fusion"]

    ov_vs_sw = bootstrap_ci_and_lift(y_b1, v4_strat_oof, baseline_pred=sw_strat_oof, n_boot=n_boot, seed=42)
    ov_vs_v2 = bootstrap_ci_and_lift(y_b1, v4_strat_oof, baseline_pred=v2_strat_oof, n_boot=n_boot, seed=42)
    ov_vs_dsw = bootstrap_ci_and_lift(y_b1, v4_strat_oof, baseline_pred=draft_sw_strat_oof, n_boot=n_boot, seed=42)
    ov_temp_vs_sw = bootstrap_ci_and_lift(y_b1, v4_temp_oof, baseline_pred=sw_temp_oof, n_boot=n_boot, seed=42)
    ov_temp_vs_v2 = bootstrap_ci_and_lift(y_b1, v4_temp_oof, baseline_pred=v2_temp_oof, n_boot=n_boot, seed=42)
    ov_standalone_cnn = bootstrap_ci_and_lift(y_b1, v4_strat_cnn, baseline_pred=sw_strat_oof, n_boot=n_boot, seed=42)
    dl_vs_sw = bootstrap_ci_and_lift(y_dl, v4_dl_oof, baseline_pred=sw_dl_oof, n_boot=n_boot, seed=42)

    # Per-position group statistical evaluation across all 4 encoders
    pos_encoders_out: dict[str, dict[str, Any]] = {}
    hypothesis_tests: list[dict[str, Any]] = [
        {"test_id": "overall_v4_vs_stopwatch_strat", "p_raw": ov_vs_sw["p_one_sided_exact"], "delta_r2": ov_vs_sw["delta_r2"], "partial_r2": ov_vs_sw["partial_r2"]},
        {"test_id": "overall_v4_vs_draft_stopwatch_strat", "p_raw": ov_vs_dsw["p_one_sided_exact"], "delta_r2": ov_vs_dsw["delta_r2"], "partial_r2": ov_vs_dsw["partial_r2"]},
        {"test_id": "overall_v4_vs_v2_strat", "p_raw": ov_vs_v2["p_one_sided_exact"], "delta_r2": ov_vs_v2["delta_r2"], "partial_r2": ov_vs_v2["partial_r2"]},
        {"test_id": "overall_v4_vs_stopwatch_temporal_loyo", "p_raw": ov_temp_vs_sw["p_one_sided_exact"], "delta_r2": ov_temp_vs_sw["delta_r2"], "partial_r2": ov_temp_vs_sw["partial_r2"]},
        {"test_id": "overall_v4_vs_v2_temporal_loyo", "p_raw": ov_temp_vs_v2["p_one_sided_exact"], "delta_r2": ov_temp_vs_v2["delta_r2"], "partial_r2": ov_temp_vs_v2["partial_r2"]},
        {"test_id": "dl_get_off_v4_vs_stopwatch", "p_raw": dl_vs_sw["p_one_sided_exact"], "delta_r2": dl_vs_sw["delta_r2"], "partial_r2": dl_vs_sw["partial_r2"]},
    ]

    pos_raw_sw_p: list[float] = []
    pos_raw_v2_p: list[float] = []
    for grp in POSITION_GROUPS:
        gm = b1_groups == grp
        yg = y_b1[gm]
        m_sw = bootstrap_ci_and_lift(yg, v4_strat_oof[gm], baseline_pred=sw_strat_oof[gm], n_boot=n_boot, seed=42)
        m_v2 = bootstrap_ci_and_lift(yg, v4_strat_oof[gm], baseline_pred=v2_strat_oof[gm], n_boot=n_boot, seed=42)
        m_dsw = bootstrap_ci_and_lift(yg, v4_strat_oof[gm], baseline_pred=draft_sw_strat_oof[gm], n_boot=n_boot, seed=42)
        m_temp = bootstrap_ci_and_lift(yg, v4_temp_oof[gm], baseline_pred=sw_temp_oof[gm], n_boot=n_boot, seed=42)
        m_cnn = bootstrap_ci_and_lift(yg, v4_strat_cnn[gm], baseline_pred=sw_strat_oof[gm], n_boot=n_boot, seed=42)
        pos_raw_sw_p.append(m_sw["p_one_sided_exact"])
        pos_raw_v2_p.append(m_v2["p_one_sided_exact"])
        hypothesis_tests.append(
            {"test_id": f"pos_{grp}_v4_vs_stopwatch", "p_raw": m_sw["p_one_sided_exact"], "delta_r2": m_sw["delta_r2"], "partial_r2": m_sw["partial_r2"]}
        )
        sal_grp = saliency_payload["cohort_position_saliency"][grp]
        pos_encoders_out[grp] = {
            "label": POSITION_GROUP_META[grp]["label"],
            "combine_positions": POSITION_GROUP_META[grp]["combine_positions"],
            "biomechanical_focus": POSITION_GROUP_META[grp]["biomechanical_focus"],
            "conv_kernels": POSITION_GROUP_META[grp]["conv_kernels"],
            "n_attn_heads": POSITION_GROUP_META[grp]["n_attn_heads"],
            "trainable_parameters": gpu_meta["trainable_parameters_per_encoder"],
            "n_players": int(gm.sum()),
            "stopwatch_r2": m_sw["baseline_r2"],
            "v2_unified_r2": m_v2["baseline_r2"],
            "v4_positional_cnn_r2": m_cnn["cv_r2"],
            "cv_r2": m_sw["cv_r2"],
            "cv_r2_ci95": m_sw["cv_r2_ci95"],
            "spearman_rho": m_sw["spearman_rho"],
            "rho_ci95": m_sw["rho_ci95"],
            "temporal_loyo_r2": m_temp["cv_r2"],
            "delta_r2_vs_stopwatch": m_sw["delta_r2"],
            "delta_r2_vs_stopwatch_ci95": m_sw["delta_r2_ci95"],
            "partial_r2_vs_stopwatch": m_sw["partial_r2"],
            "partial_r2_vs_stopwatch_ci95": m_sw["partial_r2_ci95"],
            "partial_r2_vs_draft_stopwatch": m_dsw["partial_r2"],
            "delta_r2_vs_v2": m_v2["delta_r2"],
            "p_vs_stopwatch": m_sw["p_one_sided"],
            "p_vs_v2": m_v2["p_one_sided"],
            "peak_saliency_window_sec": [sal_grp["mean_peak_window_start_sec"], sal_grp["mean_peak_window_end_sec"]],
            "mean_0_3s_attention_mass": sal_grp["mean_0_3s_attention_mass"],
            "top_mil_attended_drills": sal_grp["top_mil_attended_drills"],
        }

    # Apply Benjamini-Hochberg FDR across all hypothesis tests + per-group families
    all_q = benjamini_hochberg_fdr([h["p_raw"] for h in hypothesis_tests])
    for h, q in zip(hypothesis_tests, all_q):
        h["p_raw"] = round(float(h["p_raw"]), 4)
        h["q_fdr"] = round(float(q), 4)
        h["significant_fdr_05"] = bool(q < 0.05)

    q_pos_sw = benjamini_hochberg_fdr(pos_raw_sw_p)
    q_pos_v2 = benjamini_hochberg_fdr(pos_raw_v2_p)
    for grp, q_sw, q_v2 in zip(POSITION_GROUPS, q_pos_sw, q_pos_v2):
        pos_encoders_out[grp]["q_fdr_vs_stopwatch"] = round(float(q_sw), 4)
        pos_encoders_out[grp]["q_fdr_vs_v2"] = round(float(q_v2), 4)

    q_lookup = {h["test_id"]: h["q_fdr"] for h in hypothesis_tests}
    v2_b1_s = v2_metrics_file["modeling_results"]["benchmark_1_game_kinematic_translation"]["stratified_5fold"]
    v2_b1_t = v2_metrics_file["modeling_results"]["benchmark_1_game_kinematic_translation"]["temporal_loyo"]
    v2_b3 = v2_metrics_file["modeling_results"]["benchmark_3_dl_get_off"]["stratified_5fold"]

    payload = {
        "version": "v4-positional-attn-gpu",
        "dataset_scope": {
            "total_prospects": 510,
            "tracked_game_prospects": int(len(b1_sub)),
            "combine_10hz_player_frames": int(gpu_meta["total_combine_player_frames"]),
            "nfl_game_tracking_frames": int(v2_metrics_file["dataset_inventory"]["game_tracking_total_rows"]),
            "n_bootstrap_resamples": int(n_boot),
            "cv_protocol": "Strictly zero-leakage player-disjoint Stratified 5-Fold + Draft-Year GroupKFold Temporal LOYO (2023/2024/2025)",
        },
        "gpu_run": gpu_meta,
        "stopwatch_baseline": {
            "cv_r2": round(float(b1_s_m["stopwatch_anthro"]["r2"]), 4),
            "cv_r2_ci95": [round(float(x), 4) for x in b1_s_m["stopwatch_anthro"]["r2_ci95"]],
            "spearman_rho": round(float(b1_s_m["stopwatch_anthro"]["spearman_rho"]), 4),
            "temporal_loyo_r2": round(float(b1_t_m["stopwatch_anthro"]["r2"]), 4),
            "draft_plus_stopwatch_cv_r2": round(float(b1_s_m["draft_stopwatch"]["r2"]), 4),
        },
        "v2_baseline": {
            "cv_r2": round(float(v2_b1_s["v2_deep_fusion"]["r2"]), 4),
            "cv_r2_ci95": [round(float(x), 4) for x in v2_b1_s["v2_deep_fusion"]["r2_ci95"]],
            "spearman_rho": round(float(v2_b1_s["v2_deep_fusion"]["spearman_rho"]), 4),
            "temporal_loyo_r2": round(float(v2_b1_t["v2_deep_fusion"]["r2"]), 4),
            "standalone_unified_cnn_r2": round(float(v2_b1_s["cnn_1d_attention"]["r2"]), 4),
            "dl_get_off_r2": round(float(v2_b3["v2_deep_fusion"]["r2"]), 4),
        },
        "overall_v4_oof": {
            "n": int(len(b1_sub)),
            "cv_r2": ov_vs_sw["cv_r2"],
            "cv_r2_exact": ov_vs_sw["cv_r2_exact"],
            "cv_r2_ci95": ov_vs_sw["cv_r2_ci95"],
            "spearman_rho": ov_vs_sw["spearman_rho"],
            "rho_ci95": ov_vs_sw["rho_ci95"],
            "rmse": ov_vs_sw["rmse"],
            "standalone_v4_positional_cnn_r2": ov_standalone_cnn["cv_r2"],
            "standalone_v4_positional_cnn_rho": ov_standalone_cnn["spearman_rho"],
            "temporal_loyo_r2": ov_temp_vs_sw["cv_r2"],
            "temporal_loyo_ci95": ov_temp_vs_sw["cv_r2_ci95"],
            "temporal_loyo_rho": ov_temp_vs_sw["spearman_rho"],
            "delta_r2_vs_v2": ov_vs_v2["delta_r2"],
            "delta_r2_vs_v2_ci95": ov_vs_v2["delta_r2_ci95"],
            "p_vs_v2": ov_vs_v2["p_one_sided"],
            "q_fdr_vs_v2": q_lookup["overall_v4_vs_v2_strat"],
            "delta_r2_vs_v2_temporal": ov_temp_vs_v2["delta_r2"],
            "delta_r2_vs_stopwatch": ov_vs_sw["delta_r2"],
            "delta_r2_vs_stopwatch_ci95": ov_vs_sw["delta_r2_ci95"],
            "partial_r2_vs_stopwatch": ov_vs_sw["partial_r2"],
            "partial_r2_vs_stopwatch_ci95": ov_vs_sw["partial_r2_ci95"],
            "p_vs_stopwatch": ov_vs_sw["p_one_sided"],
            "q_fdr_vs_stopwatch": q_lookup["overall_v4_vs_stopwatch_strat"],
            "partial_r2_vs_draft_stopwatch": ov_vs_dsw["partial_r2"],
            "partial_r2_vs_draft_stopwatch_ci95": ov_vs_dsw["partial_r2_ci95"],
            "p_vs_draft_stopwatch": ov_vs_dsw["p_one_sided"],
            "q_fdr_vs_draft_stopwatch": q_lookup["overall_v4_vs_draft_stopwatch_strat"],
            "dl_get_off_v4_r2": dl_vs_sw["cv_r2"],
            "dl_get_off_v4_ci95": dl_vs_sw["cv_r2_ci95"],
            "dl_get_off_partial_r2_vs_stopwatch": dl_vs_sw["partial_r2"],
            "dl_get_off_q_fdr": q_lookup["dl_get_off_v4_vs_stopwatch"],
        },
        "positional_encoders": pos_encoders_out,
        "statistical_rigor": {
            "fdr_method": "Benjamini-Hochberg step-up",
            "bootstrap_resamples": int(n_boot),
            "hypothesis_tests": hypothesis_tests,
        },
        "eval_elapsed_sec": round(time.time() - t0, 2),
    }

    out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    out_metrics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[V4-EVAL DONE] V2 cv_r2={payload['v2_baseline']['cv_r2']:.4f} -> "
        f"V4 cv_r2={payload['overall_v4_oof']['cv_r2']:.4f} "
        f"(Temporal LOYO {payload['v2_baseline']['temporal_loyo_r2']:.4f} -> {payload['overall_v4_oof']['temporal_loyo_r2']:.4f}, "
        f"Partial R2 vs Stopwatch={payload['overall_v4_oof']['partial_r2_vs_stopwatch']:.4f}, "
        f"DL Get-Off={payload['overall_v4_oof']['dl_get_off_v4_r2']:.4f})",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL BDB 2027 V4 Position-Specific Attention GPU Pipeline")
    parser.add_argument("--mode", choices=["train", "dispatch", "evaluate"], required=True)
    parser.add_argument("--account", default=os.environ.get("COLAB_ACCOUNT", "ultra-4"))
    parser.add_argument("--gpu", default="L4")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tensors-npz", type=Path, default=DEFAULT_TENSORS_NPZ)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_V4_OOF_NPZ)
    parser.add_argument("--out-meta", type=Path, default=DEFAULT_V4_GPU_META)
    parser.add_argument("--out-metrics", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--out-saliency", type=Path, default=DEFAULT_SALIENCY_OUT)
    parser.add_argument("--pretrain-epochs", type=int, default=36)
    parser.add_argument("--specialist-epochs", type=int, default=38)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    if args.mode == "train":
        train_v4_on_gpu(
            args.tensors_npz,
            args.out_npz,
            args.out_meta,
            pretrain_epochs=args.pretrain_epochs,
            specialist_epochs=args.specialist_epochs,
        )
    elif args.mode == "dispatch":
        dispatch_v4_to_colab(
            account=args.account,
            preferred_gpu=args.gpu,
            tensors_npz=args.tensors_npz,
            out_oof_npz=args.out_npz,
            out_meta_json=args.out_meta,
        )
        evaluate_v4_locally(
            data_dir=args.data_dir,
            tensors_npz=args.tensors_npz,
            v4_oof_npz=args.out_npz,
            v4_gpu_meta=args.out_meta,
            out_metrics_json=args.out_metrics,
            out_saliency_json=args.out_saliency,
            n_boot=args.n_boot,
        )
    elif args.mode == "evaluate":
        evaluate_v4_locally(
            data_dir=args.data_dir,
            tensors_npz=args.tensors_npz,
            v4_oof_npz=args.out_npz,
            v4_gpu_meta=args.out_meta,
            out_metrics_json=args.out_metrics,
            out_saliency_json=args.out_saliency,
            n_boot=args.n_boot,
        )


if __name__ == "__main__":
    main()
