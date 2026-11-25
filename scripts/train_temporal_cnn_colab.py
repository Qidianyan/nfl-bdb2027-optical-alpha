#!/usr/bin/env python3
"""Colab GPU Multi-Scale 1D-CNN + Temporal & MIL Attention Sequence Encoder
for NFL Big Data Bowl 2027 (431,094 real 10 Hz Combine optical tracking frames).

Architecture & Strict Zero-Leakage Protocol:
1. Constructs fixed-length 5-channel 10 Hz optical sequence tensors
   `(s, a_tan, omega, kappa, jerk)` across all 510 prospects and all official
   Combine drill repetitions (`combine_tracking.csv`, 431,094 PLAYER frames).
2. Runs strictly nested zero-leakage CV aligned 1:1 with `models.py` outer splits:
   - Stratified 5-Fold CV (balanced by combine_position, random_state=42)
   - Temporal 3-Fold Leave-One-Draft-Year-Out CV (holdout 2023 / 2024 / 2025)
   Inside every outer training fold `tr_idx`:
   - Fits per-drill & global channel normalization + anthro scalers strictly on `tr_idx`.
   - Runs 5-fold inner CV × 3 seeds with Stochastic Weight Averaging (SWA) on GPU
     to emit unbiased `inner_oof` predictions/semantic neural embeddings for `tr_idx`
     and 15-model deep ensemble predictions/embeddings for `va_idx` (zero outer leakage).
3. Also extracts per-frame 10 Hz temporal attention weights on out-of-fold validation
   players for real-data Film-Room visualization.

Modes:
  --mode prepare : Build compressed 10 Hz tensor bundle (.npz) from official CSVs.
  --mode train   : Train Multi-Scale 1D-CNN + Attention on GPU (or CPU fallback) & export OOF bundle.
  --mode dispatch: End-to-end local->Colab GPU (`COLAB_ACCOUNT=ultra-4`, A100->L4 fallback, EXIT trap)->download.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local package import works when executed as a standalone script
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
SRC_DIR = CODE_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SEQ_LEN = 96  # 9.6 seconds at exact 10 Hz (>90th pct rep length without clipping)
MAX_REPS = 18  # Covers >=98% of reps per player (median=12, max=22)
N_CHANNELS = 5  # [s, a_tan, omega_rad_s, kappa, jerk]
CHANNEL_NAMES = ["s", "a_tan", "omega", "kappa", "jerk"]
COMBINE_POSITIONS = ["WR", "DB", "DL", "OL", "TE"]
POS_TO_ID = {p: i for i, p in enumerate(COMBINE_POSITIONS)}

DRILL_GROUP_MAP = {
    "FORTY_YARD_DASH": 0,
    "THREE_CONE_DRILL": 1,
    "SHORT_SHUTTLE": 2,
}  # All SKILL_DRILLS_* map to 3


def _drill_group_id(drill_type: str) -> int:
    return DRILL_GROUP_MAP.get(drill_type, 3)


def prepare_10hz_sequence_tensors(data_dir: Path, out_npz: Path) -> dict[str, Any]:
    """Build real 5-channel 10 Hz optical sequence tensors directly from official CSVs."""
    import polars as pl
    from nfl_bdb2027.data import DatasetPaths
    from nfl_bdb2027.kinematics import REP_GROUP_COLS, compute_frame_kinematics
    from nfl_bdb2027.outcomes import build_outcomes_table

    t0 = time.time()
    paths = DatasetPaths.from_dir(data_dir)
    outcomes_df = build_outcomes_table(paths).sort_values("nfl_id").reset_index(drop=True)
    nfl_ids = outcomes_df["nfl_id"].to_numpy(dtype=np.int64)
    n_players = len(nfl_ids)
    id_to_idx = {int(pid): i for i, pid in enumerate(nfl_ids)}

    ct_df = (
        compute_frame_kinematics(pl.scan_csv(paths.combine_tracking, null_values=["NA", ""], infer_schema_length=10000))
        .select(
            REP_GROUP_COLS
            + [
                "frame_idx",
                "s",
                "a_tan",
                "omega_rad_s",
                "kappa",
                "jerk",
            ]
        )
        .collect()
    )
    total_player_frames = int(ct_df.height)

    # Canonical drill dictionary (deterministic alphabetical order)
    canon_drills = sorted(ct_df["canon_drill"].unique().to_list())
    drill_to_id = {d: i for i, d in enumerate(canon_drills)}

    # Pre-allocate tensors
    seq_x = np.zeros((n_players, MAX_REPS, N_CHANNELS, SEQ_LEN), dtype=np.float32)
    frame_mask = np.zeros((n_players, MAX_REPS, SEQ_LEN), dtype=np.float32)
    rep_mask = np.zeros((n_players, MAX_REPS), dtype=np.float32)
    rep_drill_id = np.zeros((n_players, MAX_REPS), dtype=np.int64)
    rep_group_id = np.zeros((n_players, MAX_REPS), dtype=np.int64)
    rep_len_arr = np.zeros((n_players, MAX_REPS), dtype=np.int32)

    # Group frames per rep in Polars -> fast Python packing
    rep_grouped = ct_df.group_by(REP_GROUP_COLS, maintain_order=True).agg([
        pl.col("s").fill_null(0.0).alias("s_seq"),
        pl.col("a_tan").fill_null(0.0).clip(-15.0, 15.0).alias("atan_seq"),
        pl.col("omega_rad_s").fill_null(0.0).clip(0.0, 18.0).alias("omega_seq"),
        pl.col("kappa").fill_null(0.0).clip(0.0, 5.0).alias("kappa_seq"),
        pl.col("jerk").fill_null(0.0).clip(-30.0, 30.0).alias("jerk_seq"),
        pl.len().alias("n_f"),
    ])

    # Sort reps per player so high-signal timed/skill reps come first (40-yd, 3-cone, shuttle, then longest skill reps)
    rep_records_by_player: dict[int, list[dict[str, Any]]] = {i: [] for i in range(n_players)}
    for row in rep_grouped.iter_rows(named=True):
        pid = int(row["nfl_id"])
        if pid not in id_to_idx:
            continue
        p_idx = id_to_idx[pid]
        grp = _drill_group_id(str(row["drill_type"]))
        priority = 0 if grp == 0 else (1 if grp in (1, 2) else 2)
        rep_records_by_player[p_idx].append({
            "priority": priority,
            "grp": grp,
            "did": drill_to_id[str(row["canon_drill"])],
            "canon_drill": str(row["canon_drill"]),
            "n_f": int(row["n_f"]),
            "s": np.asarray(row["s_seq"], dtype=np.float32),
            "a_tan": np.asarray(row["atan_seq"], dtype=np.float32),
            "omega": np.asarray(row["omega_seq"], dtype=np.float32),
            "kappa": np.asarray(row["kappa_seq"], dtype=np.float32),
            "jerk": np.asarray(row["jerk_seq"], dtype=np.float32),
        })

    packed_reps = 0
    packed_frames = 0
    for p_idx in range(n_players):
        recs = rep_records_by_player[p_idx]
        # Deterministic ordering: timed drills first, then diverse canonical skill drills by peak mechanical energy
        recs.sort(key=lambda r: (r["priority"], -float(np.max(r["s"])), r["canon_drill"], -r["n_f"]))
        # Keep up to MAX_REPS while ensuring unique drill coverage before duplicate attempts
        chosen: list[dict[str, Any]] = []
        seen_drills: set[str] = set()
        deferred: list[dict[str, Any]] = []
        for r in recs:
            if r["canon_drill"] not in seen_drills:
                chosen.append(r)
                seen_drills.add(r["canon_drill"])
            else:
                deferred.append(r)
        chosen = (chosen + deferred)[:MAX_REPS]

        for r_i, r in enumerate(chosen):
            n_f = r["n_f"]
            arr = np.stack([r["s"], r["a_tan"], r["omega"], r["kappa"], r["jerk"]], axis=0)  # (5, n_f)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if n_f <= SEQ_LEN:
                seq_x[p_idx, r_i, :, :n_f] = arr
                frame_mask[p_idx, r_i, :n_f] = 1.0
                rep_len_arr[p_idx, r_i] = n_f
            else:
                # Uniformly resample long reps (>9.6s) to SEQ_LEN preserving start-to-end phase
                grid_src = np.linspace(0.0, n_f - 1.0, SEQ_LEN)
                xp = np.arange(n_f, dtype=np.float64)
                for c in range(N_CHANNELS):
                    seq_x[p_idx, r_i, c, :] = np.interp(grid_src, xp, arr[c]).astype(np.float32)
                frame_mask[p_idx, r_i, :] = 1.0
                rep_len_arr[p_idx, r_i] = SEQ_LEN
            rep_mask[p_idx, r_i] = 1.0
            rep_drill_id[p_idx, r_i] = r["did"]
            rep_group_id[p_idx, r_i] = r["grp"]
            packed_reps += 1
            packed_frames += n_f

    anthro_cols = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span"]
    anthro = outcomes_df[anthro_cols].to_numpy(dtype=np.float32)
    pos_id = outcomes_df["combine_position"].map(POS_TO_ID).to_numpy(dtype=np.int64)
    draft_year = outcomes_df["draft_year"].to_numpy(dtype=np.int64)
    display_names = outcomes_df["display_name"].astype(str).to_numpy()

    # Targets for B1 (game_kinematic_index + 3 components), B2 (nfl_value_index + 3 components + B1 aux), B3 (DL get-off)
    b1_targets = outcomes_df[
        ["game_kinematic_index", "z_game_p99_s", "z_game_p95_abs_a", "z_game_p95_power"]
    ].to_numpy(dtype=np.float32)
    b2_targets = outcomes_df[
        ["nfl_value_index", "z_snaps_per_season", "z_starts_per_season", "z_accolade_rate", "game_kinematic_index"]
    ].to_numpy(dtype=np.float32)
    dl_get_off = outcomes_df["pp_get_off_mean"].to_numpy(dtype=np.float32)
    dl_get_off_n = outcomes_df["pp_get_off_n"].fillna(0).to_numpy(dtype=np.int32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        nfl_ids=nfl_ids,
        display_names=display_names,
        pos_id=pos_id,
        draft_year=draft_year,
        anthro=anthro,
        seq_x=seq_x,
        frame_mask=frame_mask,
        rep_mask=rep_mask,
        rep_drill_id=rep_drill_id,
        rep_group_id=rep_group_id,
        rep_len=rep_len_arr,
        canon_drills=np.asarray(canon_drills),
        b1_targets=b1_targets,
        b2_targets=b2_targets,
        dl_get_off=dl_get_off,
        dl_get_off_n=dl_get_off_n,
        total_player_frames=np.int64(total_player_frames),
        packed_reps=np.int64(packed_reps),
        packed_frames=np.int64(packed_frames),
    )
    elapsed = time.time() - t0
    meta = {
        "n_players": n_players,
        "max_reps": MAX_REPS,
        "seq_len": SEQ_LEN,
        "n_channels": N_CHANNELS,
        "channels": CHANNEL_NAMES,
        "n_canon_drills": len(canon_drills),
        "total_player_frames_csv": total_player_frames,
        "packed_reps": packed_reps,
        "packed_frames": packed_frames,
        "bundle_mb": round(out_npz.stat().st_size / (1024 * 1024), 2),
        "prepare_sec": round(elapsed, 2),
    }
    print(f"[PREPARE] Built {out_npz} ({meta['bundle_mb']} MB) in {elapsed:.1f}s | "
          f"{total_player_frames:,} CSV frames -> {packed_reps:,} reps across {n_players} players.")
    return meta


# ---------------------------------------------------------------------------
# PyTorch Multi-Scale 1D-CNN + Temporal & MIL Attention Sequence Encoder
# ---------------------------------------------------------------------------

def _build_torch_model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SqueezeExcite1D(nn.Module):
        def __init__(self, channels: int, reduction: int = 4) -> None:
            super().__init__()
            hid = max(8, channels // reduction)
            self.fc1 = nn.Linear(channels, hid)
            self.fc2 = nn.Linear(hid, channels)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # x: (N, C, L), mask: (N, 1, L)
            denom = mask.sum(dim=-1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=-1) / denom
            gate = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))).unsqueeze(-1)
            return x * gate

    class MultiScale1DCNNTemporalAttention(nn.Module):
        """Hierarchical 10 Hz sequence encoder:
        1. Multi-Scale 1D-CNN stem (k=3 [0.3s], k=7 [0.7s], k=15 [1.5s]) on
           raw-normalized + drill-relative 5-channel physics sequences + time phase.
        2. Residual SE-Conv1D refinement + FiLM drill conditioning.
        3. Masked Gated Temporal Attention Pooling + Masked Peak Pooling over 96 frames.
        4. Masked Player-Level Cross-Rep MIL Attention Aggregator over up to 18 reps
           (seamlessly handles 40-yd & agility opt-outs via attention masking).
        5. Multi-Task Projection Head with semantically aligned outputs.
        """

        def __init__(
            self,
            in_channels: int,
            n_drills: int,
            n_targets: int,
            d_branch: int = 16,
            d_rep: int = 48,
            d_player: int = 36,
            dropout: float = 0.28,
        ) -> None:
            super().__init__()
            d_conv = d_branch * 3  # 48
            self.conv_k3 = nn.Conv1d(in_channels, d_branch, kernel_size=3, padding=1)
            self.conv_k7 = nn.Conv1d(in_channels, d_branch, kernel_size=7, padding=3)
            self.conv_k15 = nn.Conv1d(in_channels, d_branch, kernel_size=15, padding=7)
            self.gn1 = nn.GroupNorm(6, d_conv)
            self.drop1 = nn.Dropout(dropout)

            self.conv_res = nn.Conv1d(d_conv, d_conv, kernel_size=5, padding=2)
            self.gn2 = nn.GroupNorm(6, d_conv)
            self.se = SqueezeExcite1D(d_conv, reduction=4)
            self.drop2 = nn.Dropout(dropout)

            # Drill FiLM conditioning on sequence features
            self.drill_emb = nn.Embedding(n_drills + 1, 16)
            self.group_emb = nn.Embedding(4, 8)
            self.film = nn.Linear(24, d_conv * 2)

            # Temporal Gated Attention over L=96 frames
            self.t_attn_v = nn.Linear(d_conv, 24)
            self.t_attn_u = nn.Linear(d_conv, 24)
            self.t_attn_w = nn.Linear(24, 1)

            # Rep summary projection: [attn_pool(48), max_pool(48), mean_pool(48), drill(16), group(8)]
            self.rep_proj = nn.Sequential(
                nn.Linear(d_conv * 3 + 24, d_rep),
                nn.LayerNorm(d_rep),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Player-level MIL Attention across up to R=18 drill reps
            self.mil_v = nn.Linear(d_rep, 24)
            self.mil_u = nn.Linear(d_rep, 24)
            self.mil_w = nn.Linear(24, 1)

            # Context fusion (position embedding + standardized anthropometrics)
            self.pos_emb = nn.Embedding(len(COMBINE_POSITIONS), 8)
            self.anthro_proj = nn.Sequential(
                nn.Linear(5, 12),
                nn.LayerNorm(12),
                nn.GELU(),
            )

            # Player bottleneck (combining MIL-attended reps + mean reps + skill-only reps + context)
            self.player_head = nn.Sequential(
                nn.Linear(d_rep * 2 + 8 + 12, d_player),
                nn.LayerNorm(d_player),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_all = nn.Linear(d_player, n_targets)

            # Dedicated Skill-Drill-Only branch (100% coverage even for 40-yd opt-outs)
            self.skill_head = nn.Sequential(
                nn.Linear(d_rep + 8 + 12, 24),
                nn.LayerNorm(24),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(24, 1),
            )

        def forward(
            self,
            x: torch.Tensor,          # (B, R, C_in, L)
            fmask: torch.Tensor,      # (B, R, L)
            rmask: torch.Tensor,      # (B, R)
            drill_id: torch.Tensor,   # (B, R)
            group_id: torch.Tensor,   # (B, R)
            pos_id: torch.Tensor,     # (B,)
            anthro: torch.Tensor,     # (B, 5)
            return_attn: bool = False,
        ) -> tuple[torch.Tensor, ...]:
            B, R, C, L = x.shape
            xf = x.reshape(B * R, C, L)
            mf = fmask.reshape(B * R, 1, L)

            h = torch.cat(
                [self.conv_k3(xf), self.conv_k7(xf), self.conv_k15(xf)],
                dim=1,
            )
            h = self.drop1(F.gelu(self.gn1(h))) * mf
            h_res = self.drop2(F.gelu(self.gn2(self.conv_res(h)))) * mf
            h = self.se(h + h_res, mf) * mf

            d_emb = self.drill_emb(drill_id.reshape(B * R))
            g_emb = self.group_emb(group_id.reshape(B * R))
            dg = torch.cat([d_emb, g_emb], dim=-1)
            gamma, beta = self.film(dg).chunk(2, dim=-1)
            h = (h * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1)) + 0.25 * beta.unsqueeze(-1)) * mf

            # Temporal Gated Attention over L frames
            ht = h.transpose(1, 2)  # (BR, L, D)
            t_logits = self.t_attn_w(torch.tanh(self.t_attn_v(ht)) * torch.sigmoid(self.t_attn_u(ht))).squeeze(-1)
            t_logits = t_logits.masked_fill(mf.squeeze(1) < 0.5, -1e4)
            t_weights = torch.softmax(t_logits, dim=-1) * mf.squeeze(1)
            t_weights = t_weights / t_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            attn_pool = torch.bmm(t_weights.unsqueeze(1), ht).squeeze(1)
            mean_pool = (h * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1.0)
            masked_for_max = h.masked_fill(mf < 0.5, -1e4)
            max_pool = masked_for_max.max(dim=-1).values
            max_pool = torch.where(mf.sum(dim=-1) > 0.5, max_pool, torch.zeros_like(max_pool))

            rep_feat = self.rep_proj(torch.cat([attn_pool, max_pool, mean_pool, dg], dim=-1))
            rep_feat = rep_feat.reshape(B, R, -1)

            # Player-level MIL Attention over valid reps
            m_logits = self.mil_w(torch.tanh(self.mil_v(rep_feat)) * torch.sigmoid(self.mil_u(rep_feat))).squeeze(-1)
            m_logits = m_logits.masked_fill(rmask < 0.5, -1e4)
            m_weights = torch.softmax(m_logits, dim=-1) * rmask
            m_weights = m_weights / m_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            p_attn = torch.bmm(m_weights.unsqueeze(1), rep_feat).squeeze(1)
            p_mean = (rep_feat * rmask.unsqueeze(-1)).sum(dim=1) / rmask.sum(dim=1, keepdim=True).clamp(min=1.0)

            # Attention entropy (normalized scalar summarizing kinematic consistency across drills)
            ent = -(m_weights * (m_weights.clamp(min=1e-6)).log()).sum(dim=-1, keepdim=True) / math.log(MAX_REPS)

            # Skill-drill-only aggregate (group_id == 3)
            sk_mask = rmask * (group_id == 3).float()
            sk_denom = sk_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            p_skill = (rep_feat * sk_mask.unsqueeze(-1)).sum(dim=1) / sk_denom

            p_ctx = torch.cat([self.pos_emb(pos_id), self.anthro_proj(anthro)], dim=-1)
            player_z = self.player_head(torch.cat([p_attn, p_mean, p_ctx], dim=-1))
            preds_all = self.out_all(player_z)
            pred_skill = self.skill_head(torch.cat([p_skill, p_ctx], dim=-1))

            if return_attn:
                return preds_all, pred_skill, ent, t_weights.reshape(B, R, L), m_weights
            return preds_all, pred_skill, ent

    return MultiScale1DCNNTemporalAttention


def _fit_fold_channel_stats(
    seq_x: np.ndarray,       # (N, R, 5, L)
    fmask: np.ndarray,       # (N, R, L)
    rmask: np.ndarray,       # (N, R)
    drill_id: np.ndarray,    # (N, R)
    n_drills: int,
    tr_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit global and EB-shrunk per-drill channel mean/std strictly on `tr_idx`."""
    g_mean = np.zeros(N_CHANNELS, dtype=np.float32)
    g_std = np.ones(N_CHANNELS, dtype=np.float32)
    d_mean = np.zeros((n_drills, N_CHANNELS), dtype=np.float32)
    d_std = np.ones((n_drills, N_CHANNELS), dtype=np.float32)

    sub_x = seq_x[tr_idx]      # (Nt, R, 5, L)
    sub_fm = fmask[tr_idx]     # (Nt, R, L)
    sub_rm = rmask[tr_idx]     # (Nt, R)
    sub_did = drill_id[tr_idx] # (Nt, R)

    valid_all = (sub_fm > 0.5) & (sub_rm[:, :, None] > 0.5)
    for c in range(N_CHANNELS):
        vals = sub_x[:, :, c, :][valid_all]
        gm = float(np.mean(vals)) if len(vals) > 0 else 0.0
        gs = float(np.std(vals)) if len(vals) > 10 and np.std(vals) > 1e-4 else 1.0
        g_mean[c] = gm
        g_std[c] = gs
        d_mean[:, c] = gm
        d_std[:, c] = gs

    shrink_k = 120.0
    for d in range(n_drills):
        d_valid = valid_all & (sub_did[:, :, None] == d)
        n_obs = float(d_valid.sum())
        if n_obs < 10:
            continue
        for c in range(N_CHANNELS):
            vals = sub_x[:, :, c, :][d_valid]
            dm = float(np.mean(vals))
            ds = max(float(np.std(vals)), 1e-3)
            d_mean[d, c] = (n_obs * dm + shrink_k * g_mean[c]) / (n_obs + shrink_k)
            d_std[d, c] = (n_obs * ds + shrink_k * g_std[c]) / (n_obs + shrink_k)
    return g_mean, g_std, d_mean, d_std


def _build_fold_input_tensor(
    seq_x: np.ndarray,
    fmask: np.ndarray,
    rep_len: np.ndarray,
    drill_id: np.ndarray,
    g_mean: np.ndarray,
    g_std: np.ndarray,
    d_mean: np.ndarray,
    d_std: np.ndarray,
) -> np.ndarray:
    """Create 11-channel input tensor per rep:
    - 5 globally standardized physical channels (s, a_tan, omega, kappa, jerk)
    - 5 drill-relative EB-standardized channels (captures within-drill dominance)
    - 1 normalized temporal phase channel t_rel in [0, 1]
    """
    N, R, _, L = seq_x.shape
    out = np.zeros((N, R, N_CHANNELS * 2 + 1, L), dtype=np.float32)

    # Global z channels
    gz = (seq_x - g_mean[None, None, :, None]) / g_std[None, None, :, None]
    out[:, :, :N_CHANNELS, :] = np.clip(gz, -5.0, 5.0)

    # Drill-relative z channels
    dm = d_mean[drill_id][:, :, :, None]  # (N, R, 5, 1)
    ds = d_std[drill_id][:, :, :, None]
    dz = (seq_x - dm) / ds
    out[:, :, N_CHANNELS : 2 * N_CHANNELS, :] = np.clip(dz, -5.0, 5.0)

    # Normalized elapsed time phase in [0, 1]
    t_idx = np.arange(L, dtype=np.float32)[None, None, :]
    denom = np.maximum(rep_len[:, :, None].astype(np.float32) - 1.0, 1.0)
    t_rel = np.clip(t_idx / denom, 0.0, 1.0)
    out[:, :, 2 * N_CHANNELS, :] = t_rel

    # Zero out padding strictly
    out *= fmask[:, :, None, :]
    return out


def _fit_transform_anthro(
    anthro: np.ndarray,
    pos_id: np.ndarray,
    tr_idx: np.ndarray,
) -> np.ndarray:
    """Empirical-Bayes positional anthro standardization strictly on `tr_idx`."""
    n_p, n_f = anthro.shape
    out = np.zeros((n_p, n_f), dtype=np.float32)
    shrink_k = 10.0
    for j in range(n_f):
        col_tr = anthro[tr_idx, j]
        fin = col_tr[np.isfinite(col_tr)]
        gm = float(np.mean(fin)) if len(fin) > 0 else 0.0
        gs = float(np.std(fin)) if len(fin) > 1 and np.std(fin) > 1e-5 else 1.0
        pm = np.full(len(COMBINE_POSITIONS), gm, dtype=np.float32)
        ps = np.full(len(COMBINE_POSITIONS), gs, dtype=np.float32)
        for p in range(len(COMBINE_POSITIONS)):
            vals = col_tr[pos_id[tr_idx] == p]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                n_obs = float(len(vals))
                pm[p] = (n_obs * float(np.mean(vals)) + shrink_k * gm) / (n_obs + shrink_k)
                s_raw = float(np.std(vals)) if len(vals) > 1 and np.std(vals) > 1e-5 else gs
                ps[p] = max((n_obs * s_raw + shrink_k * gs) / (n_obs + shrink_k), 1e-4)
        raw = anthro[:, j]
        m_p = pm[pos_id]
        s_p = ps[pos_id]
        filled = np.where(np.isfinite(raw), raw, m_p)
        out[:, j] = np.clip((filled - m_p) / s_p, -4.5, 4.5)
    return out


def _train_single_gpu_model(
    ModelCls: Any,
    x_tr: "torch.Tensor",
    fm_tr: "torch.Tensor",
    rm_tr: "torch.Tensor",
    did_tr: "torch.Tensor",
    gid_tr: "torch.Tensor",
    pid_tr: "torch.Tensor",
    an_tr: "torch.Tensor",
    y_tr: "torch.Tensor",
    x_va: "torch.Tensor",
    fm_va: "torch.Tensor",
    rm_va: "torch.Tensor",
    did_va: "torch.Tensor",
    gid_va: "torch.Tensor",
    pid_va: "torch.Tensor",
    an_va: "torch.Tensor",
    n_drills: int,
    seed: int,
    epochs: int = 70,
    batch_size: int = 64,
    lr: float = 2.2e-3,
    weight_decay: float = 3.0e-2,
    return_attn_va: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Train one Multi-Scale 1D-CNN + Attention encoder with SWA and predict on validation."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = x_tr.device
    n_tr = x_tr.shape[0]
    n_targets = y_tr.shape[1]

    model = ModelCls(
        in_channels=x_tr.shape[2],
        n_drills=n_drills,
        n_targets=n_targets,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # Target scaling on primary & auxiliary columns strictly from y_tr
    y_mask_all = torch.isfinite(y_tr)
    y_safe = torch.where(y_mask_all, y_tr, torch.zeros_like(y_tr))
    y_means = (y_safe * y_mask_all).sum(dim=0) / y_mask_all.sum(dim=0).clamp(min=1.0)
    y_vars = (((y_safe - y_means) ** 2) * y_mask_all).sum(dim=0) / y_mask_all.sum(dim=0).clamp(min=1.0)
    y_stds = y_vars.sqrt().clamp(min=1e-4)
    y_norm = torch.where(y_mask_all, (y_tr - y_means) / y_stds, torch.zeros_like(y_tr))

    swa_start = max(1, epochs - 14)
    swa_state: dict[str, torch.Tensor] | None = None
    swa_count = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 4:
                continue
            bx = x_tr[idx].clone()
            bfm = fm_tr[idx]
            brm = rm_tr[idx].clone()
            bdid = did_tr[idx]
            bgid = gid_tr[idx]
            bpid = pid_tr[idx]
            ban = an_tr[idx]
            by = y_norm[idx]
            bym = y_mask_all[idx]

            # Physics-safe 10 Hz data augmentation: mild sensor jitter + random rep dropout (simulates opt-outs!)
            noise = torch.randn_like(bx[:, :, : N_CHANNELS * 2, :]) * 0.04
            bx[:, :, : N_CHANNELS * 2, :] = (bx[:, :, : N_CHANNELS * 2, :] + noise) * bfm.unsqueeze(2)
            if torch.rand(1, device=device).item() < 0.7:
                keep = (torch.rand_like(brm) > 0.14).float()
                # Guarantee at least 3 reps remain per player
                brm_aug = brm * keep
                too_few = brm_aug.sum(dim=1, keepdim=True) < 3.0
                brm = torch.where(too_few, brm, brm_aug)

            # Mild temporal shift (+/- 1 frame) on 35% of batches
            if torch.rand(1, device=device).item() < 0.35:
                shift = 1 if torch.rand(1, device=device).item() < 0.5 else -1
                bx = torch.roll(bx, shifts=shift, dims=-1) * bfm.unsqueeze(2)

            preds_all, pred_skill, _ = model(bx, bfm, brm, bdid, bgid, bpid, ban)

            # Primary target loss: Huber + Cosine-Correlation ranking alignment
            p0 = preds_all[:, 0]
            t0 = by[:, 0]
            loss_huber = F.smooth_l1_loss(p0, t0, beta=0.6)
            p0c = p0 - p0.mean()
            t0c = t0 - t0.mean()
            corr = (p0c * t0c).sum() / (p0c.norm().clamp(min=1e-5) * t0c.norm().clamp(min=1e-5))
            loss_primary = 0.72 * loss_huber + 0.28 * (1.0 - corr)

            # Skill-drill-only auxiliary head (forces 100%-coverage skill drills to carry full predictive power)
            loss_skill = F.smooth_l1_loss(pred_skill.squeeze(-1), t0, beta=0.6)

            # Multi-task auxiliary heads
            loss_aux = torch.tensor(0.0, device=device)
            if n_targets > 1:
                aux_m = bym[:, 1:]
                if aux_m.any():
                    aux_diff = F.smooth_l1_loss(preds_all[:, 1:], by[:, 1:], reduction="none", beta=0.6)
                    loss_aux = (aux_diff * aux_m.float()).sum() / aux_m.float().sum().clamp(min=1.0)

            loss = loss_primary + 0.35 * loss_aux + 0.25 * loss_skill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()
        scheduler.step()

        if ep >= swa_start:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    for k, v in model.state_dict().items():
                        swa_state[k].add_(v.detach())
                swa_count += 1

    if swa_state is not None and swa_count > 0:
        for k in swa_state:
            if swa_state[k].is_floating_point():
                swa_state[k].div_(float(swa_count))
        model.load_state_dict(swa_state)

    model.eval()
    with torch.no_grad():
        if return_attn_va:
            p_all, p_sk, ent, t_w, m_w = model(
                x_va, fm_va, rm_va, did_va, gid_va, pid_va, an_va, return_attn=True
            )
            t_attn_np = t_w.cpu().numpy()
            m_attn_np = m_w.cpu().numpy()
        else:
            p_all, p_sk, ent = model(x_va, fm_va, rm_va, did_va, gid_va, pid_va, an_va, return_attn=False)
            t_attn_np, m_attn_np = None, None

        # Rescale back to original target units
        p_all_orig = p_all * y_stds.unsqueeze(0) + y_means.unsqueeze(0)
        p_sk_orig = p_sk * y_stds[0] + y_means[0]
        feat_out = torch.cat([p_all_orig, p_sk_orig, ent], dim=-1).cpu().numpy()
    return feat_out, t_attn_np, m_attn_np


def _build_outer_splits(
    pos_id: np.ndarray,
    draft_year: np.ndarray,
    mode: str,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Exact 1:1 split alignment with `nfl_bdb2027.models.build_cv_splits`."""
    from sklearn.model_selection import StratifiedKFold

    if mode == "stratified":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return [
            (tr, va, f"fold_{i + 1}")
            for i, (tr, va) in enumerate(skf.split(np.zeros(len(pos_id)), pos_id))
        ]
    if mode == "temporal":
        splits = []
        for yr in sorted(np.unique(draft_year)):
            tr = np.where(draft_year != yr)[0]
            va = np.where(draft_year == yr)[0]
            splits.append((tr, va, f"holdout_{int(yr)}"))
        return splits
    raise ValueError(f"Unsupported mode: {mode}")


def _run_nested_benchmark(
    ModelCls: Any,
    device: "torch.device",
    name: str,
    subset_mask: np.ndarray,
    seq_x_all: np.ndarray,
    fmask_all: np.ndarray,
    rmask_all: np.ndarray,
    rep_len_all: np.ndarray,
    drill_id_all: np.ndarray,
    group_id_all: np.ndarray,
    pos_id_all: np.ndarray,
    draft_year_all: np.ndarray,
    anthro_all: np.ndarray,
    targets_all: np.ndarray,
    n_drills: int,
    cv_mode: str,
    seeds: tuple[int, ...] = (42, 1024),
    epochs: int = 70,
    inner_folds: int = 5,
    capture_attn: bool = False,
) -> dict[str, Any]:
    """Run strictly zero-leakage nested outer+inner CV on GPU.

    Returns per-outer-fold `inner_oof_feats` (for leak-free meta-stacking in models.py),
    outer `va_feats`, assembled full `oof_feats`, and standalone OOF R^2 / Spearman rho.
    """
    import torch
    from scipy import stats
    from sklearn.metrics import r2_score
    from sklearn.model_selection import StratifiedKFold

    sub_indices = np.where(subset_mask)[0]
    seq_x = seq_x_all[sub_indices]
    fmask = fmask_all[sub_indices]
    rmask = rmask_all[sub_indices]
    rep_len = rep_len_all[sub_indices]
    drill_id = drill_id_all[sub_indices]
    group_id = group_id_all[sub_indices]
    pos_id = pos_id_all[sub_indices]
    draft_year = draft_year_all[sub_indices]
    anthro = anthro_all[sub_indices]
    targets = targets_all[sub_indices]

    n_sub = len(sub_indices)
    n_targets = targets.shape[1]
    out_dim = n_targets + 2  # [primary, aux_1..aux_k, skill_only_pred, attn_entropy]

    outer_splits = _build_outer_splits(pos_id, draft_year, mode=cv_mode, seed=42)
    oof_feats = np.zeros((n_sub, out_dim), dtype=np.float32)
    oof_t_attn = np.zeros((n_sub, MAX_REPS, SEQ_LEN), dtype=np.float32) if capture_attn else None
    oof_m_attn = np.zeros((n_sub, MAX_REPS), dtype=np.float32) if capture_attn else None

    fold_inner_oof: list[np.ndarray] = []
    fold_va_feats: list[np.ndarray] = []
    fold_tr_indices: list[np.ndarray] = []
    fold_va_indices: list[np.ndarray] = []

    t0 = time.time()
    for fold_i, (tr_idx, va_idx, fold_name) in enumerate(outer_splits):
        # Strictly fit channel and anthro normalization ONLY on outer tr_idx!
        gm, gs, dm, ds = _fit_fold_channel_stats(seq_x, fmask, rmask, drill_id, n_drills, tr_idx)
        x_norm = _build_fold_input_tensor(seq_x, fmask, rep_len, drill_id, gm, gs, dm, ds)
        an_norm = _fit_transform_anthro(anthro, pos_id, tr_idx)

        # Move fold tensors to GPU once
        x_tr_t = torch.from_numpy(x_norm[tr_idx]).to(device)
        fm_tr_t = torch.from_numpy(fmask[tr_idx]).to(device)
        rm_tr_t = torch.from_numpy(rmask[tr_idx]).to(device)
        did_tr_t = torch.from_numpy(drill_id[tr_idx]).to(device)
        gid_tr_t = torch.from_numpy(group_id[tr_idx]).to(device)
        pid_tr_t = torch.from_numpy(pos_id[tr_idx]).to(device)
        an_tr_t = torch.from_numpy(an_norm[tr_idx]).to(device)
        y_tr_t = torch.from_numpy(targets[tr_idx]).to(device)

        x_va_t = torch.from_numpy(x_norm[va_idx]).to(device)
        fm_va_t = torch.from_numpy(fmask[va_idx]).to(device)
        rm_va_t = torch.from_numpy(rmask[va_idx]).to(device)
        did_va_t = torch.from_numpy(drill_id[va_idx]).to(device)
        gid_va_t = torch.from_numpy(group_id[va_idx]).to(device)
        pid_va_t = torch.from_numpy(pos_id[va_idx]).to(device)
        an_va_t = torch.from_numpy(an_norm[va_idx]).to(device)

        # Inner CV strictly inside outer tr_idx (produces unbiased inner_oof + deep ensemble on va_idx)
        tr_pos = pos_id[tr_idx]
        min_cls = int(np.bincount(tr_pos)[np.bincount(tr_pos) > 0].min())
        n_in = min(inner_folds, max(2, min_cls))
        if len(np.unique(tr_pos)) > 1:
            in_splits = list(StratifiedKFold(n_splits=n_in, shuffle=True, random_state=42 + fold_i * 17).split(tr_idx, tr_pos))
        else:
            from sklearn.model_selection import KFold
            in_splits = list(KFold(n_splits=n_in, shuffle=True, random_state=42 + fold_i * 17).split(tr_idx))

        inner_oof = np.zeros((len(tr_idx), out_dim), dtype=np.float32)
        va_ens = np.zeros((len(va_idx), out_dim), dtype=np.float32)
        va_t_acc = np.zeros((len(va_idx), MAX_REPS, SEQ_LEN), dtype=np.float32) if capture_attn else None
        va_m_acc = np.zeros((len(va_idx), MAX_REPS), dtype=np.float32) if capture_attn else None
        n_va_models = 0

        for in_i, (itr, iva) in enumerate(in_splits):
            itr_t = torch.from_numpy(itr).to(device)
            iva_t = torch.from_numpy(iva).to(device)
            seed_iva_acc = np.zeros((len(iva), out_dim), dtype=np.float32)
            for s_i, sd in enumerate(seeds):
                run_seed = sd + fold_i * 101 + in_i * 13
                # Predict both iva and outer va_idx in one pass
                x_eval = torch.cat([x_tr_t[iva_t], x_va_t], dim=0)
                fm_eval = torch.cat([fm_tr_t[iva_t], fm_va_t], dim=0)
                rm_eval = torch.cat([rm_tr_t[iva_t], rm_va_t], dim=0)
                did_eval = torch.cat([did_tr_t[iva_t], did_va_t], dim=0)
                gid_eval = torch.cat([gid_tr_t[iva_t], gid_va_t], dim=0)
                pid_eval = torch.cat([pid_tr_t[iva_t], pid_va_t], dim=0)
                an_eval = torch.cat([an_tr_t[iva_t], an_va_t], dim=0)

                pred_eval, t_at, m_at = _train_single_gpu_model(
                    ModelCls,
                    x_tr_t[itr_t],
                    fm_tr_t[itr_t],
                    rm_tr_t[itr_t],
                    did_tr_t[itr_t],
                    gid_tr_t[itr_t],
                    pid_tr_t[itr_t],
                    an_tr_t[itr_t],
                    y_tr_t[itr_t],
                    x_eval,
                    fm_eval,
                    rm_eval,
                    did_eval,
                    gid_eval,
                    pid_eval,
                    an_eval,
                    n_drills=n_drills,
                    seed=run_seed,
                    epochs=epochs,
                    return_attn_va=capture_attn,
                )
                n_iva = len(iva)
                seed_iva_acc += pred_eval[:n_iva]
                va_ens += pred_eval[n_iva:]
                if capture_attn and t_at is not None and m_at is not None:
                    va_t_acc += t_at[n_iva:]
                    va_m_acc += m_at[n_iva:]
                n_va_models += 1
            inner_oof[iva] = seed_iva_acc / float(len(seeds))

        va_ens /= float(n_va_models)
        oof_feats[va_idx] = va_ens
        if capture_attn and oof_t_attn is not None and oof_m_attn is not None:
            oof_t_attn[va_idx] = va_t_acc / float(n_va_models)
            oof_m_attn[va_idx] = va_m_acc / float(n_va_models)

        fold_inner_oof.append(inner_oof)
        fold_va_feats.append(va_ens)
        fold_tr_indices.append(tr_idx)
        fold_va_indices.append(va_idx)

    y_true = targets[:, 0]
    y_pred = oof_feats[:, 0]
    r2 = float(r2_score(y_true, y_pred))
    rho = float(stats.spearmanr(y_true, y_pred)[0])
    skill_r2 = float(r2_score(y_true, oof_feats[:, -2]))
    skill_rho = float(stats.spearmanr(y_true, oof_feats[:, -2])[0])
    elapsed = time.time() - t0
    print(
        f"  [{name:22s} | {cv_mode:10s}] n={n_sub:3d} | CNN-Attn R2={r2:+.4f}, rho={rho:+.4f} "
        f"| SkillOnly R2={skill_r2:+.4f}, rho={skill_rho:+.4f} ({elapsed:.1f}s)"
    )

    return {
        "sub_indices": sub_indices,
        "oof_feats": oof_feats,
        "oof_t_attn": oof_t_attn,
        "oof_m_attn": oof_m_attn,
        "fold_inner_oof": fold_inner_oof,
        "fold_va_feats": fold_va_feats,
        "fold_tr_indices": fold_tr_indices,
        "fold_va_indices": fold_va_indices,
        "summary": {
            "n": int(n_sub),
            "r2": round(r2, 5),
            "spearman_rho": round(rho, 5),
            "skill_only_r2": round(skill_r2, 5),
            "skill_only_rho": round(skill_rho, 5),
            "elapsed_sec": round(elapsed, 2),
        },
    }


def train_on_gpu(data_npz: Path, out_npz: Path, out_meta_json: Path, epochs: int = 72) -> dict[str, Any]:
    """Train all zero-leakage nested Multi-Scale 1D-CNN + Attention models on GPU."""
    import torch

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    print(f"[TRAIN] PyTorch {torch.__version__} | Device: {device} ({gpu_name}, {vram_gb} GB VRAM)")

    d = np.load(data_npz, allow_pickle=True)
    nfl_ids = d["nfl_ids"]
    pos_id = d["pos_id"]
    draft_year = d["draft_year"]
    anthro = d["anthro"]
    seq_x = d["seq_x"]
    fmask = d["frame_mask"]
    rmask = d["rep_mask"]
    rep_len = d["rep_len"]
    drill_id = d["rep_drill_id"]
    group_id = d["rep_group_id"]
    canon_drills = d["canon_drills"]
    b1_targets = d["b1_targets"]
    b2_targets = d["b2_targets"]
    dl_get_off = d["dl_get_off"]
    dl_get_off_n = d["dl_get_off_n"]
    n_drills = len(canon_drills)

    ModelCls = _build_torch_model_classes()
    # Count model parameters for paper documentation
    probe_m = ModelCls(in_channels=N_CHANNELS * 2 + 1, n_drills=n_drills, n_targets=4)
    n_params = sum(p.numel() for p in probe_m.parameters() if p.requires_grad)
    print(f"[MODEL] MultiScale1DCNNTemporalAttention | trainable_params={n_params:,} | "
          f"tensor={seq_x.shape} | total_csv_frames={int(d['total_player_frames']):,}")

    # Benchmark 1: Sunday Game Kinematic Translation (n=497 where game_kinematic_index is finite)
    b1_mask = np.isfinite(b1_targets[:, 0])
    b1_strat = _run_nested_benchmark(
        ModelCls, device, "B1_game_kin", b1_mask,
        seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id, draft_year, anthro,
        b1_targets, n_drills, cv_mode="stratified", seeds=(42, 1024, 2027), epochs=epochs,
        capture_attn=True,
    )
    b1_temp = _run_nested_benchmark(
        ModelCls, device, "B1_game_kin", b1_mask,
        seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id, draft_year, anthro,
        b1_targets, n_drills, cv_mode="temporal", seeds=(42, 1024, 2027), epochs=epochs,
        capture_attn=False,
    )

    # Benchmark 2: Exposure-Normalized NFL Career Value (n=510)
    b2_mask = np.isfinite(b2_targets[:, 0])
    b2_strat = _run_nested_benchmark(
        ModelCls, device, "B2_nfl_value", b2_mask,
        seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id, draft_year, anthro,
        b2_targets, n_drills, cv_mode="stratified", seeds=(42, 1024, 2027), epochs=epochs,
        capture_attn=False,
    )
    b2_temp = _run_nested_benchmark(
        ModelCls, device, "B2_nfl_value", b2_mask,
        seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id, draft_year, anthro,
        b2_targets, n_drills, cv_mode="temporal", seeds=(42, 1024, 2027), epochs=epochs,
        capture_attn=False,
    )

    # Benchmark 3: DL/EDGE Regular-Season First-Step Get-Off (n=115)
    dl_mask = (pos_id == POS_TO_ID["DL"]) & (dl_get_off_n >= 15) & np.isfinite(dl_get_off)
    b3_targets = np.stack([dl_get_off, b1_targets[:, 0]], axis=1).astype(np.float32)
    b3_strat = _run_nested_benchmark(
        ModelCls, device, "B3_dl_get_off", dl_mask,
        seq_x, fmask, rmask, rep_len, drill_id, group_id, pos_id, draft_year, anthro,
        b3_targets, n_drills, cv_mode="stratified", seeds=(42, 1024, 2027), epochs=epochs,
        capture_attn=False,
    )

    # Pack everything cleanly into a portable .npz for zero-leakage local fusion in models.py
    save_dict: dict[str, Any] = {
        "nfl_ids": nfl_ids,
        "canon_drills": canon_drills,
        "rep_drill_id": drill_id,
        "rep_group_id": group_id,
        "rep_len": rep_len,
        "b1_strat_oof": b1_strat["oof_feats"],
        "b1_strat_sub_idx": b1_strat["sub_indices"],
        "b1_strat_t_attn": b1_strat["oof_t_attn"],
        "b1_strat_m_attn": b1_strat["oof_m_attn"],
        "b1_temp_oof": b1_temp["oof_feats"],
        "b1_temp_sub_idx": b1_temp["sub_indices"],
        "b2_strat_oof": b2_strat["oof_feats"],
        "b2_strat_sub_idx": b2_strat["sub_indices"],
        "b2_temp_oof": b2_temp["oof_feats"],
        "b2_temp_sub_idx": b2_temp["sub_indices"],
        "b3_strat_oof": b3_strat["oof_feats"],
        "b3_strat_sub_idx": b3_strat["sub_indices"],
    }

    for tag, bundle in [
        ("b1_strat", b1_strat),
        ("b1_temp", b1_temp),
        ("b2_strat", b2_strat),
        ("b2_temp", b2_temp),
        ("b3_strat", b3_strat),
    ]:
        save_dict[f"{tag}_n_folds"] = np.int64(len(bundle["fold_inner_oof"]))
        for fi in range(len(bundle["fold_inner_oof"])):
            save_dict[f"{tag}_f{fi}_inner_oof"] = bundle["fold_inner_oof"][fi]
            save_dict[f"{tag}_f{fi}_va_feats"] = bundle["fold_va_feats"][fi]
            save_dict[f"{tag}_f{fi}_tr_idx"] = bundle["fold_tr_indices"][fi]
            save_dict[f"{tag}_f{fi}_va_idx"] = bundle["fold_va_indices"][fi]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **save_dict)

    total_sec = round(time.time() - t_start, 2)
    meta = {
        "architecture": "Multi-Scale 1D-CNN (k=3,7,15) + SE-ResBlock + FiLM Drill Conditioning + Gated Temporal & Cross-Rep MIL Attention",
        "trainable_parameters": int(n_params),
        "input_channels": CHANNEL_NAMES,
        "derived_input_channels": 11,
        "sequence_length_frames": SEQ_LEN,
        "sampling_hz": 10.0,
        "max_reps_per_player": MAX_REPS,
        "total_combine_player_frames": int(d["total_player_frames"]),
        "packed_reps": int(d["packed_reps"]),
        "device": str(device),
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": str(torch.__version__),
        "epochs_per_fold": epochs,
        "seeds_per_inner_fold": [42, 1024, 2027],
        "nested_cv_protocol": "Strict outer-fold normalization + 5-fold inner CV x 3 seeds SWA (zero outer leakage)",
        "total_gpu_train_sec": total_sec,
        "standalone_cnn_oof": {
            "b1_stratified_5fold": b1_strat["summary"],
            "b1_temporal_loyo": b1_temp["summary"],
            "b2_stratified_5fold": b2_strat["summary"],
            "b2_temporal_loyo": b2_temp["summary"],
            "b3_dl_get_off_stratified": b3_strat["summary"],
        },
    }
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[DONE] Saved GPU OOF bundle to {out_npz} & metadata to {out_meta_json} in {total_sec:.1f}s.")
    return meta


def dispatch_to_colab(
    data_dir: Path,
    artifacts_dir: Path,
    account: str = "ultra-4",
    preferred_gpu: str = "L4",
    epochs: int = 72,
) -> None:
    """Prepare 10 Hz tensors locally, dispatch to Colab GPU with strict EXIT trap,
    download OOF artifacts, and verify zero active sessions / 0.00/hr billing."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    local_tensors = artifacts_dir / "nfl_10hz_tensors.npz"
    local_oof = artifacts_dir / "colab_temporal_cnn_oof.npz"
    local_meta = artifacts_dir / "colab_gpu_run_metadata.json"

    prepare_10hz_sequence_tensors(data_dir, local_tensors)

    runner_sh = SCRIPT_DIR / "run_colab_gpu_job.sh"
    env = os.environ.copy()
    env["COLAB_ACCOUNT"] = account
    env["PREFERRED_GPU"] = preferred_gpu
    env["EPOCHS"] = str(epochs)
    env["LOCAL_TENSORS"] = str(local_tensors)
    env["LOCAL_OOF"] = str(local_oof)
    env["LOCAL_META"] = str(local_meta)
    env["TRAIN_SCRIPT"] = str(Path(__file__).resolve())

    subprocess.run(["bash", str(runner_sh)], env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NFL BDB 2027 Multi-Scale 1D-CNN + Attention Temporal Encoder (Colab GPU)"
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "dispatch"],
        default="train",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Volumes/Extra/datasets/kaggle-nfl2027/nfl-big-data-bowl-2027"),
    )
    parser.add_argument(
        "--tensors-npz",
        type=Path,
        default=CODE_ROOT / "artifacts" / "nfl_10hz_tensors.npz",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=CODE_ROOT / "artifacts" / "colab_temporal_cnn_oof.npz",
    )
    parser.add_argument(
        "--out-meta",
        type=Path,
        default=CODE_ROOT / "artifacts" / "colab_gpu_run_metadata.json",
    )
    parser.add_argument("--account", type=str, default=os.environ.get("COLAB_ACCOUNT", "ultra-4"))
    parser.add_argument("--gpu", type=str, default="L4")
    parser.add_argument("--epochs", type=int, default=72)
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_10hz_sequence_tensors(args.data_dir, args.tensors_npz)
    elif args.mode == "train":
        train_on_gpu(args.tensors_npz, args.out_npz, args.out_meta, epochs=args.epochs)
    elif args.mode == "dispatch":
        dispatch_to_colab(
            data_dir=args.data_dir,
            artifacts_dir=args.out_npz.parent,
            account=args.account,
            preferred_gpu=args.gpu,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
