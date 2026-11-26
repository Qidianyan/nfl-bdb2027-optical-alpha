"""V4 Position-Specific Multi-Scale 1D-CNN + Explicit Dual-Head Temporal Attention,
Residual Meta-Fusion, and Rigorous Statistical Inference (BH-FDR, Partial R^2,
2,000x Paired Bootstrap 95% CI, Zero-Leakage Group Splits).

Runs natively in pure NumPy/SciPy/scikit-learn for local inference, testing, and
meta-fusion, while exposing the exact PyTorch CUDA encoder (`build_torch_positional_encoder`)
when PyTorch is installed (e.g., on Google Colab NVIDIA L4/A100 GPU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from .config import COMBINE_POSITIONS, DT_SEC
except ImportError:
    COMBINE_POSITIONS = ["WR", "DB", "DL", "OL", "TE"]
    DT_SEC = 0.1

# Canonical 4 disjoint, exhaustive position groups covering all 510 NFL BDB 2027 prospects.
# Note: Official BDB 2027 combine_results.csv uses 5 combine groups (WR, DB, DL, OL, TE);
# edge/OLB rushers appear under DL, so OL_LB covers OL trench/box athletes cleanly with zero overlap.
POSITION_GROUPS: tuple[str, ...] = ("WR_TE", "DB", "DL_EDGE", "OL_LB")

POSITION_GROUP_META: dict[str, dict[str, Any]] = {
    "WR_TE": {
        "combine_positions": ["WR", "TE"],
        "pos_ids": [0, 4],
        "label": "Pass Catchers (WR / TE)",
        "biomechanical_focus": "Route-stem deceleration braking, curvilinear cut force (s*omega), and out-of-break YAC re-acceleration",
        "conv_kernels": [3, 5, 9],
        "n_attn_heads": 2,
    },
    "DB": {
        "combine_positions": ["DB"],
        "pos_ids": [1],
        "label": "Defensive Backs (CB / S)",
        "biomechanical_focus": "Backpedal hip-flip angular transition, 90-degree break deceleration, and 0-1.0s closing burst",
        "conv_kernels": [3, 5, 9],
        "n_attn_heads": 2,
    },
    "DL_EDGE": {
        "combine_positions": ["DL"],
        "pos_ids": [2],
        "label": "Defensive Line & Edge (DL / EDGE)",
        "biomechanical_focus": "0-10 yd explosive launch slope, mass-scaled mechanical power (m*v*a), and pass-rush first-step get-off",
        "conv_kernels": [3, 5, 9],
        "n_attn_heads": 2,
    },
    "OL_LB": {
        "combine_positions": ["OL"],
        "pos_ids": [3],
        "label": "Offensive Line / Trench Box (OL)",
        "biomechanical_focus": "Short-area lateral mirror stability, pull/wave linear impulse (m*a_max), and low-speed balance retention",
        "conv_kernels": [3, 5, 9],
        "n_attn_heads": 2,
    },
}

COMBINE_POS_TO_GROUP: dict[str, str] = {
    "WR": "WR_TE",
    "TE": "WR_TE",
    "DB": "DB",
    "DL": "DL_EDGE",
    "OL": "OL_LB",
    "LB": "OL_LB",
    "EDGE": "DL_EDGE",
}

POS_ID_TO_GROUP: dict[int, str] = {
    0: "WR_TE",    # WR
    1: "DB",       # DB
    2: "DL_EDGE",  # DL
    3: "OL_LB",    # OL
    4: "WR_TE",    # TE
}

GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(POSITION_GROUPS)}

ENRICHED_CHANNEL_NAMES: tuple[str, ...] = (
    "s_glob_z",
    "a_tan_glob_z",
    "omega_glob_z",
    "kappa_glob_z",
    "jerk_glob_z",
    "s_drill_z",
    "a_tan_drill_z",
    "omega_drill_z",
    "kappa_drill_z",
    "jerk_drill_z",
    "a_cent_z",
    "pos_power_z",
    "braking_z",
    "t_rel",
)
N_ENRICHED_CHANNELS = len(ENRICHED_CHANNEL_NAMES)  # 14


def map_combine_position_to_group(combine_pos: str) -> str:
    """Map an official NFL Combine position string to one of the 4 canonical V4 position groups."""
    key = str(combine_pos).strip().upper()
    if key not in COMBINE_POS_TO_GROUP:
        raise ValueError(f"Unsupported combine_position '{combine_pos}'; expected one of {sorted(COMBINE_POS_TO_GROUP)}")
    return COMBINE_POS_TO_GROUP[key]


def map_pos_id_to_group(pos_id: int) -> str:
    """Map integer combine_position ID (0..4) to one of the 4 canonical V4 position groups."""
    pid = int(pos_id)
    if pid not in POS_ID_TO_GROUP:
        raise ValueError(f"Unsupported pos_id {pos_id}; expected 0..4")
    return POS_ID_TO_GROUP[pid]


# ---------------------------------------------------------------------------
# Strict Within-Fold 14-Channel 10 Hz Physics Enrichment
# ---------------------------------------------------------------------------

def fit_fold_channel_stats_v4(
    seq_x: np.ndarray,       # (N, R, 5, L) -> [s, a_tan, omega, kappa, jerk]
    fmask: np.ndarray,       # (N, R, L)
    rmask: np.ndarray,       # (N, R)
    drill_id: np.ndarray,    # (N, R)
    n_drills: int,
    tr_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit global, within-drill, and derived biomechanical channel scalers strictly on `tr_idx`."""
    sx = np.nan_to_num(seq_x[tr_idx], nan=0.0, posinf=0.0, neginf=0.0)
    fm = np.nan_to_num(fmask[tr_idx], nan=0.0, posinf=0.0, neginf=0.0)
    rm = np.nan_to_num(rmask[tr_idx], nan=0.0, posinf=0.0, neginf=0.0)
    did = drill_id[tr_idx]

    valid = (fm > 0.5) & (rm[:, :, None] > 0.5)  # (N_tr, R, L)

    # Raw 5 channels global mean/std
    glob_mean = np.zeros(5, dtype=np.float32)
    glob_std = np.ones(5, dtype=np.float32)
    for c in range(5):
        vals = sx[:, :, c, :][valid]
        if vals.size > 10:
            glob_mean[c] = float(np.mean(vals))
            glob_std[c] = max(float(np.std(vals)), 1e-4)

    # Derived 3 biomechanical channels: a_cent = s*omega, pos_power = s*max(a_tan,0), braking = max(-a_tan,0)
    s_ch = np.clip(sx[:, :, 0, :], 0.0, 14.0)
    a_ch = np.clip(sx[:, :, 1, :], -15.0, 15.0)
    om_ch = np.clip(sx[:, :, 2, :], 0.0, 18.0)
    derived = np.stack(
        [
            s_ch * om_ch,
            s_ch * np.maximum(a_ch, 0.0),
            np.maximum(-a_ch, 0.0),
        ],
        axis=2,
    )
    der_mean = np.zeros(3, dtype=np.float32)
    der_std = np.ones(3, dtype=np.float32)
    for j in range(3):
        vals = derived[:, :, j, :][valid]
        if vals.size > 10:
            der_mean[j] = float(np.mean(vals))
            der_std[j] = max(float(np.std(vals)), 1e-4)

    # Within-drill Empirical-Bayes shrunk stats for the 5 base channels
    drill_mean = np.broadcast_to(glob_mean[None, :], (n_drills + 1, 5)).copy()
    drill_std = np.broadcast_to(glob_std[None, :], (n_drills + 1, 5)).copy()
    shrink_k = 40.0
    for d in range(n_drills):
        d_sel = valid & (did[:, :, None] == d)
        n_v = int(d_sel.sum())
        if n_v >= 15:
            for c in range(5):
                vals = sx[:, :, c, :][d_sel]
                m_raw = float(np.mean(vals))
                s_raw = max(float(np.std(vals)), 1e-4)
                drill_mean[d, c] = (n_v * m_raw + shrink_k * glob_mean[c]) / (n_v + shrink_k)
                drill_std[d, c] = max((n_v * s_raw + shrink_k * glob_std[c]) / (n_v + shrink_k), 1e-4)

    return {
        "glob_mean": glob_mean,
        "glob_std": glob_std,
        "drill_mean": drill_mean,
        "drill_std": drill_std,
        "der_mean": der_mean,
        "der_std": der_std,
    }


def build_enriched_10hz_channels(
    seq_x: np.ndarray,       # (N, R, 5, L)
    fmask: np.ndarray,       # (N, R, L)
    rep_len: np.ndarray,     # (N, R)
    drill_id: np.ndarray,    # (N, R)
    stats_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Transform raw 5-channel 10 Hz waveforms into 14 NaN-free physics-enriched channels."""
    sx = np.nan_to_num(seq_x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    fm = np.nan_to_num(fmask.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    N, R, _, L = sx.shape

    gm = stats_dict["glob_mean"]
    gs = stats_dict["glob_std"]
    dm = stats_dict["drill_mean"]
    ds = stats_dict["drill_std"]
    derm = stats_dict["der_mean"]
    ders = stats_dict["der_std"]

    out = np.zeros((N, R, N_ENRICHED_CHANNELS, L), dtype=np.float32)

    # 0..4: global z-scored raw channels
    for c in range(5):
        out[:, :, c, :] = np.clip((sx[:, :, c, :] - gm[c]) / gs[c], -5.0, 5.0) * fm

    # 5..9: drill-relative z-scored channels
    did_clip = np.clip(drill_id, 0, dm.shape[0] - 1)
    d_m = dm[did_clip]  # (N, R, 5)
    d_s = ds[did_clip]
    for c in range(5):
        out[:, :, 5 + c, :] = np.clip((sx[:, :, c, :] - d_m[:, :, c, None]) / d_s[:, :, c, None], -5.0, 5.0) * fm

    # 10..12: derived physics (centripetal cut force, positive explosive power, braking impulse)
    s_ch = np.clip(sx[:, :, 0, :], 0.0, 14.0)
    a_ch = np.clip(sx[:, :, 1, :], -15.0, 15.0)
    om_ch = np.clip(sx[:, :, 2, :], 0.0, 18.0)
    der = [
        s_ch * om_ch,
        s_ch * np.maximum(a_ch, 0.0),
        np.maximum(-a_ch, 0.0),
    ]
    for j, arr in enumerate(der):
        out[:, :, 10 + j, :] = np.clip((arr - derm[j]) / ders[j], -5.0, 5.0) * fm

    # 13: normalized elapsed phase t_rel in [0, 1]
    t_grid = np.arange(L, dtype=np.float32)[None, None, :]
    denom = np.maximum(rep_len[:, :, None].astype(np.float32) - 1.0, 1.0)
    out[:, :, 13, :] = np.clip(t_grid / denom, 0.0, 1.0) * fm
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Pure-NumPy Reference Multi-Scale 1D-CNN + Explicit Temporal Attention Layer
# ---------------------------------------------------------------------------

def _safe_masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically stable masked softmax along the last axis.
    Guarantees:
    - Zero weight strictly on padded frames (`mask < 0.5`).
    - Exact `sum(weights, axis=-1) == 1.0` on every sequence with >= 1 valid frame.
    - All-zeros (zero NaNs) on completely masked/empty sequences.
    """
    clean_logits = np.nan_to_num(logits.astype(np.float64), nan=0.0, posinf=20.0, neginf=-20.0)
    clean_mask = (np.nan_to_num(mask.astype(np.float64), nan=0.0) >= 0.5)
    temp = max(float(temperature), 1e-4)
    scaled = np.where(clean_mask, clean_logits / temp, -1e9)
    max_v = np.max(scaled, axis=-1, keepdims=True)
    max_v = np.where(np.isfinite(max_v) & (max_v > -1e8), max_v, 0.0)
    exp_v = np.where(clean_mask, np.exp(np.clip(scaled - max_v, -60.0, 60.0)), 0.0)
    denom = np.sum(exp_v, axis=-1, keepdims=True)
    has_valid = denom > 1e-12
    weights = np.where(has_valid, exp_v / np.where(has_valid, denom, 1.0), 0.0)
    return weights.astype(np.float32)


def _conv1d_same_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Exact same-padding 1D convolution in pure NumPy.
    x: (N, C_in, L), weight: (C_out, C_in, K), bias: (C_out,) -> (N, C_out, L).
    """
    x_clean = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    N, C_in, L = x_clean.shape
    C_out, w_cin, K = weight.shape
    assert C_in == w_cin, f"Channel mismatch: {C_in} vs {w_cin}"
    pad = K // 2
    xp = np.pad(x_clean, ((0, 0), (0, 0), (pad, pad)), mode="constant")
    # Build sliding windows (N, C_in, L, K) using stride views
    windows = np.lib.stride_tricks.sliding_window_view(xp, window_shape=K, axis=-1)  # (N, C_in, L, K)
    out = np.einsum("nclk,ock->nol", windows, weight, optimize=True) + bias[None, :, None]
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class TemporalAttentionPool1D:
    """Explicit single-head or dual-head gated temporal attention pooling layer (NumPy).

    Produces per-frame normalized saliency weights `alpha_t` where:
      - `alpha_t.sum(axis=-1) == 1.0` for every sequence with at least 1 valid frame,
      - `alpha_t == 0.0` for every padded frame (`mask < 0.5`),
      - NaN / Inf inputs are sanitized so outputs never contain NaN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 2,
        d_hidden: int = 24,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        if n_heads not in (1, 2):
            raise ValueError(f"n_heads must be 1 or 2, got {n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_hidden = int(d_hidden)
        self.temperature = float(temperature)
        rng = np.random.default_rng(seed)
        scale_in = math.sqrt(2.0 / (d_model + d_hidden))
        scale_out = math.sqrt(1.0 / d_hidden)
        self.w_v = rng.normal(0.0, scale_in, size=(n_heads, d_model, d_hidden)).astype(np.float32)
        self.b_v = np.zeros((n_heads, d_hidden), dtype=np.float32)
        self.w_u = rng.normal(0.0, scale_in, size=(n_heads, d_model, d_hidden)).astype(np.float32)
        self.b_u = np.zeros((n_heads, d_hidden), dtype=np.float32)
        self.w_q = rng.normal(0.0, scale_out, size=(n_heads, d_hidden)).astype(np.float32)
        self.b_q = np.zeros((n_heads,), dtype=np.float32)

    def forward(
        self,
        h: np.ndarray,      # (N, D, L)
        mask: np.ndarray,   # (N, L)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (pooled: (N, n_heads*D), alpha_combined: (N, L), alpha_heads: (N, n_heads, L))."""
        h_clean = np.nan_to_num(h.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        m_clean = np.nan_to_num(mask.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        ht = np.transpose(h_clean, (0, 2, 1))  # (N, L, D)
        N, L, D = ht.shape

        head_weights = np.zeros((N, self.n_heads, L), dtype=np.float32)
        pooled_heads = []
        for hi in range(self.n_heads):
            v = np.tanh(ht @ self.w_v[hi] + self.b_v[hi])
            u = 1.0 / (1.0 + np.exp(-np.clip(ht @ self.w_u[hi] + self.b_u[hi], -30.0, 30.0)))
            logits = (v * u) @ self.w_q[hi] + self.b_q[hi]  # (N, L)
            w = _safe_masked_softmax(logits, m_clean, temperature=self.temperature)
            head_weights[:, hi, :] = w
            pooled_heads.append(np.einsum("nl,nld->nd", w, ht))

        # Mean across heads preserves exact sum == 1.0 on every valid sequence
        alpha_combined = np.mean(head_weights, axis=1).astype(np.float32)
        # Re-normalize defensively to eliminate floating-point rounding drift
        sums = alpha_combined.sum(axis=-1, keepdims=True)
        valid_seq = sums > 1e-12
        alpha_combined = np.where(valid_seq, alpha_combined / np.where(valid_seq, sums, 1.0), 0.0).astype(np.float32)
        pooled = np.concatenate(pooled_heads, axis=-1).astype(np.float32)
        return pooled, alpha_combined, head_weights


class MultiScalePositionalEncoderNumpy:
    """Position-specific lightweight Multi-Scale 1D-CNN (kernels 3, 5, 9) +
    Explicit Dual-Head Temporal Attention Encoder (pure NumPy reference engine)."""

    def __init__(
        self,
        position_group: str,
        in_channels: int = N_ENRICHED_CHANNELS,
        d_branch: int = 12,
        n_heads: int = 2,
        kernels: tuple[int, int, int] = (3, 5, 9),
        seed: int = 42,
    ) -> None:
        if position_group not in POSITION_GROUPS:
            raise ValueError(f"Unknown position_group '{position_group}'; expected one of {POSITION_GROUPS}")
        self.position_group = position_group
        self.in_channels = int(in_channels)
        self.d_branch = int(d_branch)
        self.kernels = tuple(int(k) for k in kernels)
        self.d_model = d_branch * len(self.kernels)
        self.n_heads = int(n_heads)

        rng = np.random.default_rng(seed + GROUP_TO_IDX[position_group] * 97)
        self.conv_weights: list[np.ndarray] = []
        self.conv_biases: list[np.ndarray] = []
        for k in self.kernels:
            std = math.sqrt(2.0 / (in_channels * k))
            self.conv_weights.append(rng.normal(0.0, std, size=(d_branch, in_channels, k)).astype(np.float32))
            self.conv_biases.append(np.zeros((d_branch,), dtype=np.float32))

        self.attn_pool = TemporalAttentionPool1D(
            d_model=self.d_model,
            n_heads=n_heads,
            d_hidden=24,
            seed=seed + GROUP_TO_IDX[position_group] * 31,
        )
        self.head_w = rng.normal(0.0, math.sqrt(1.0 / (self.d_model * n_heads)), size=(self.d_model * n_heads, 1)).astype(np.float32)
        self.head_b = np.zeros((1,), dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,      # (N, C_in, L)
        mask: np.ndarray,   # (N, L)
    ) -> dict[str, np.ndarray]:
        """Run forward pass and return prediction, embedding, and exact frame saliency weights."""
        x_clean = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        m_clean = np.nan_to_num(mask.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        branches = [
            _conv1d_same_numpy(x_clean, w, b)
            for w, b in zip(self.conv_weights, self.conv_biases)
        ]
        h = np.concatenate(branches, axis=1)  # (N, d_model, L)
        # GELU activation (fast tanh approximation) + frame mask
        h = 0.5 * h * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * (h ** 3))))
        h = h * (m_clean[:, None, :] >= 0.5)

        pooled, alpha_t, alpha_heads = self.attn_pool.forward(h, m_clean)
        pred = (pooled @ self.head_w + self.head_b).squeeze(-1)
        return {
            "pred": np.nan_to_num(pred.astype(np.float32), nan=0.0),
            "embedding": np.nan_to_num(pooled.astype(np.float32), nan=0.0),
            "alpha_t": alpha_t,
            "alpha_heads": alpha_heads,
        }


# ---------------------------------------------------------------------------
# PyTorch CUDA Position-Specific Multi-Scale 1D-CNN + Dual-Head Attention
# ---------------------------------------------------------------------------

def build_torch_positional_encoder(
    in_channels: int = N_ENRICHED_CHANNELS,
    n_drills: int = 50,
    n_targets: int = 4,
    d_branch: int = 16,
    d_rep: int = 48,
    d_player: int = 36,
    n_heads: int = 2,
    dropout: float = 0.28,
) -> Any:
    """Build PyTorch `PositionalMultiScaleAttnNet` (kernels 3, 5, 9 + explicit Dual-Head
    Temporal Attention + attention-weighted physics readout + cross-rep MIL pooling)."""
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
            denom = mask.sum(dim=-1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=-1) / denom
            gate = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled)))).unsqueeze(-1)
            return x * gate

    class PositionalMultiScaleAttnNet(nn.Module):
        """V4 Position-Specific Multi-Scale 1D-CNN (k=3, 5, 9) + Dual-Head Temporal Attention.
        Head 1 captures explosive launch / peak stride burst; Head 2 captures braking /
        curvilinear change-of-direction transitions. Frame saliency `alpha_t` is the exact
        mean across heads, strictly normalized to sum(alpha_t) == 1.0 over valid frames.
        """

        def __init__(self) -> None:
            super().__init__()
            self.n_heads = int(n_heads)
            d_conv = d_branch * 3  # 48 (kernels 3, 5, 9)
            self.conv_k3 = nn.Conv1d(in_channels, d_branch, kernel_size=3, padding=1)
            self.conv_k5 = nn.Conv1d(in_channels, d_branch, kernel_size=5, padding=2)
            self.conv_k9 = nn.Conv1d(in_channels, d_branch, kernel_size=9, padding=4)
            self.gn1 = nn.GroupNorm(6, d_conv)
            self.drop1 = nn.Dropout(dropout)

            self.conv_res = nn.Conv1d(d_conv, d_conv, kernel_size=5, padding=2)
            self.gn2 = nn.GroupNorm(6, d_conv)
            self.se = SqueezeExcite1D(d_conv, reduction=4)
            self.drop2 = nn.Dropout(dropout)

            # Drill + drill-group FiLM conditioning
            self.drill_emb = nn.Embedding(n_drills + 1, 16)
            self.group_emb = nn.Embedding(4, 8)
            self.film = nn.Linear(24, d_conv * 2)

            # Explicit Dual-Head Gated Temporal Attention (outputs 2 heads per 0.1s frame)
            self.t_attn_v = nn.Linear(d_conv, 24 * self.n_heads)
            self.t_attn_u = nn.Linear(d_conv, 24 * self.n_heads)
            self.t_attn_w = nn.Parameter(torch.empty(self.n_heads, 24))
            self.t_attn_b = nn.Parameter(torch.zeros(self.n_heads))
            nn.init.xavier_uniform_(self.t_attn_w)

            # Rep projection: [head_attn_pools(n_heads*d_conv), max_pool(d_conv), mean_pool(d_conv), dg(24), attn_phys(4)]
            rep_in = d_conv * (self.n_heads + 2) + 24 + 4
            self.rep_proj = nn.Sequential(
                nn.Linear(rep_in, d_rep),
                nn.LayerNorm(d_rep),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Cross-Rep Masked MIL Attention over up to 18 reps
            self.mil_v = nn.Linear(d_rep, 24)
            self.mil_u = nn.Linear(d_rep, 24)
            self.mil_w = nn.Linear(24, 1)

            # Anthropometric + sub-position context
            self.pos_emb = nn.Embedding(len(COMBINE_POSITIONS), 8)
            self.anthro_proj = nn.Sequential(
                nn.Linear(5, 12),
                nn.LayerNorm(12),
                nn.GELU(),
            )

            # Player bottleneck combining MIL-attended reps + mean reps + attn-weighted physics (4) + context (20)
            self.player_head = nn.Sequential(
                nn.Linear(d_rep * 2 + 4 + 20, d_player),
                nn.LayerNorm(d_player),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.out_all = nn.Linear(d_player, n_targets)

            # Skill-drill-only branch (100% opt-out resilient)
            self.skill_head = nn.Sequential(
                nn.Linear(d_rep + 4 + 20, 24),
                nn.LayerNorm(24),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(24, 1),
            )

        def forward(
            self,
            x: torch.Tensor,          # (B, R, 14, L)
            fmask: torch.Tensor,      # (B, R, L)
            rmask: torch.Tensor,      # (B, R)
            drill_id: torch.Tensor,   # (B, R)
            group_id: torch.Tensor,   # (B, R)
            pos_id: torch.Tensor,     # (B,)
            anthro: torch.Tensor,     # (B, 5)
            return_attn: bool = False,
        ) -> tuple[torch.Tensor, ...]:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fmask = torch.nan_to_num(fmask, nan=0.0, posinf=0.0, neginf=0.0)
            rmask = torch.nan_to_num(rmask, nan=0.0, posinf=0.0, neginf=0.0)

            B, R, C, L = x.shape
            xf = x.reshape(B * R, C, L)
            mf = (fmask.reshape(B * R, 1, L) >= 0.5).float()

            # Multi-scale 1D-CNN stem (k=3, 5, 9)
            h = torch.cat([self.conv_k3(xf), self.conv_k5(xf), self.conv_k9(xf)], dim=1)
            h = self.drop1(F.gelu(self.gn1(h))) * mf
            h_res = self.drop2(F.gelu(self.gn2(self.conv_res(h)))) * mf
            h = self.se(h + h_res, mf) * mf

            # Drill FiLM conditioning
            d_emb = self.drill_emb(drill_id.reshape(B * R))
            g_emb = self.group_emb(group_id.reshape(B * R))
            dg = torch.cat([d_emb, g_emb], dim=-1)
            gamma, beta = self.film(dg).chunk(2, dim=-1)
            h = (h * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(-1)) + 0.25 * beta.unsqueeze(-1)) * mf

            # Explicit Dual-Head Gated Temporal Attention over L frames
            ht = h.transpose(1, 2)  # (BR, L, D)
            BR = B * R
            v = torch.tanh(self.t_attn_v(ht)).reshape(BR, L, self.n_heads, 24)
            u = torch.sigmoid(self.t_attn_u(ht)).reshape(BR, L, self.n_heads, 24)
            # (BR, n_heads, L)
            t_logits = torch.einsum("blhd,hd->bhl", v * u, self.t_attn_w) + self.t_attn_b[None, :, None]
            valid_1d = mf.squeeze(1)  # (BR, L)
            has_any = valid_1d.sum(dim=-1, keepdim=True) > 0.5  # (BR, 1)
            t_logits = t_logits.masked_fill(valid_1d.unsqueeze(1) < 0.5, -1e4)
            t_heads = torch.softmax(t_logits, dim=-1) * valid_1d.unsqueeze(1)
            t_heads = t_heads / t_heads.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            t_heads = torch.where(has_any.unsqueeze(1), t_heads, torch.zeros_like(t_heads))

            # Combined frame saliency alpha_t (BR, L): exact sum == 1.0 on every valid rep
            alpha_t = t_heads.mean(dim=1)
            alpha_t = alpha_t / alpha_t.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            alpha_t = torch.where(has_any, alpha_t, torch.zeros_like(alpha_t))

            # Attention-pooled latent features for each head -> (BR, n_heads * D)
            attn_pools = torch.bmm(t_heads, ht).reshape(BR, self.n_heads * h.shape[1])
            mean_pool = (h * mf).sum(dim=-1) / mf.sum(dim=-1).clamp(min=1.0)
            masked_max = h.masked_fill(mf < 0.5, -1e4).max(dim=-1).values
            max_pool = torch.where(has_any, masked_max, torch.zeros_like(masked_max))

            # Attention-weighted kinematic summary over channels [s_glob_z(0), a_cent_z(10), pos_power_z(11), braking_z(12)]
            phys_sel = xf[:, [0, 10, 11, 12], :].transpose(1, 2)  # (BR, L, 4)
            attn_phys = torch.bmm(alpha_t.unsqueeze(1), phys_sel).squeeze(1)  # (BR, 4)

            rep_feat = self.rep_proj(torch.cat([attn_pools, max_pool, mean_pool, dg, attn_phys], dim=-1))
            rep_feat = rep_feat.reshape(B, R, -1)
            attn_phys_r = attn_phys.reshape(B, R, 4)

            # Player-level Masked MIL Attention across up to R=18 drill reps
            m_logits = self.mil_w(torch.tanh(self.mil_v(rep_feat)) * torch.sigmoid(self.mil_u(rep_feat))).squeeze(-1)
            m_logits = m_logits.masked_fill(rmask < 0.5, -1e4)
            m_weights = torch.softmax(m_logits, dim=-1) * rmask
            m_weights = m_weights / m_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

            p_attn = torch.bmm(m_weights.unsqueeze(1), rep_feat).squeeze(1)
            p_mean = (rep_feat * rmask.unsqueeze(-1)).sum(dim=1) / rmask.sum(dim=1, keepdim=True).clamp(min=1.0)
            p_phys = torch.bmm(m_weights.unsqueeze(1), attn_phys_r).squeeze(1)  # (B, 4)

            # Normalized attention entropy across valid reps
            ent = -(m_weights * m_weights.clamp(min=1e-6).log()).sum(dim=-1, keepdim=True) / math.log(max(R, 2))

            # Skill-drill-only aggregate (group_id == 3)
            sk_mask = rmask * (group_id == 3).float()
            sk_denom = sk_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            p_skill = (rep_feat * sk_mask.unsqueeze(-1)).sum(dim=1) / sk_denom
            p_skill_phys = (attn_phys_r * sk_mask.unsqueeze(-1)).sum(dim=1) / sk_denom

            p_ctx = torch.cat([self.pos_emb(pos_id), self.anthro_proj(anthro)], dim=-1)
            player_z = self.player_head(torch.cat([p_attn, p_mean, p_phys, p_ctx], dim=-1))
            preds_all = self.out_all(player_z)
            pred_skill = self.skill_head(torch.cat([p_skill, p_skill_phys, p_ctx], dim=-1))

            if return_attn:
                return (
                    preds_all,
                    pred_skill,
                    ent,
                    p_phys,
                    alpha_t.reshape(B, R, L),
                    t_heads.reshape(B, R, self.n_heads, L),
                    m_weights,
                )
            return preds_all, pred_skill, ent, p_phys

    return PositionalMultiScaleAttnNet()


# ---------------------------------------------------------------------------
# 0.1s Frame-Level Saliency Window Extraction
# ---------------------------------------------------------------------------

def extract_peak_saliency_window(
    alpha_t: np.ndarray,
    frame_mask: np.ndarray,
    window_frames: int = 3,
    dt_sec: float = DT_SEC,
) -> dict[str, Any]:
    """Identify the contiguous `window_frames` (default 3 frames = 0.3s) window capturing
    the highest cumulative temporal attention mass over valid frames (supports arbitrary masks)."""
    a = np.nan_to_num(np.asarray(alpha_t, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    m = np.nan_to_num(np.asarray(frame_mask, dtype=np.float64), nan=0.0) >= 0.5
    if a.shape != m.shape or a.ndim != 1:
        raise ValueError(f"alpha_t and frame_mask must be matching 1D arrays, got {a.shape} vs {m.shape}")
    n_valid = int(m.sum())
    if n_valid == 0:
        return {
            "start_frame": 0,
            "end_frame": 0,
            "start_sec": 0.0,
            "end_sec": 0.0,
            "attention_mass": 0.0,
            "peak_frame": 0,
            "peak_sec": 0.0,
            "peak_weight": 0.0,
        }
    masked_a = np.where(m, np.maximum(a, 0.0), 0.0)
    L = len(masked_a)
    w = max(1, min(int(window_frames), L))
    # Prefer fully valid contiguous windows when any exist; otherwise evaluate windows containing >= 1 valid frame
    best_start = -1
    best_mass = -1.0
    for require_full in (True, False):
        for st in range(0, L - w + 1):
            win_m = m[st : st + w]
            if require_full and not bool(np.all(win_m)):
                continue
            if not require_full and not bool(np.any(win_m)):
                continue
            mass = float(masked_a[st : st + w].sum())
            if mass > best_mass:
                best_mass = mass
                best_start = st
        if best_start >= 0:
            break
    if best_start < 0:
        best_start = int(np.argmax(m))
        w = 1
        best_mass = float(masked_a[best_start])
    peak_scores = np.where(m, masked_a, -1.0)
    peak_frame = int(np.argmax(peak_scores))
    return {
        "start_frame": int(best_start),
        "end_frame": int(best_start + w - 1),
        "start_sec": round(best_start * dt_sec, 2),
        "end_sec": round((best_start + w) * dt_sec, 2),
        "attention_mass": round(float(best_mass), 4),
        "peak_frame": peak_frame,
        "peak_sec": round(peak_frame * dt_sec, 2),
        "peak_weight": round(float(masked_a[peak_frame]), 4),
    }


# ---------------------------------------------------------------------------
# Rigorous Statistical Inference: BH-FDR, Partial R^2, 2,000x Bootstrap CI
# ---------------------------------------------------------------------------

def benjamini_hochberg_fdr(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Compute Benjamini-Hochberg step-up FDR-adjusted q-values.

    Guarantees:
    - Monotonicity with respect to sorted p-values (`q[order]` is non-decreasing).
    - `0.0 <= p_i <= q_i <= 1.0` for all valid `p_i in [0, 1]`.
    - Handles NaN/Inf gracefully by clamping invalid p-values to 1.0.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.array([], dtype=np.float64)
    p_clean = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 1.0)
    order = np.argsort(p_clean, kind="mergesort")
    ranked_p = p_clean[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw_q = ranked_p * (float(n) / ranks)
    # Enforce monotonicity from largest rank downward
    adj_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    q_out = np.empty(n, dtype=np.float64)
    q_out[order] = adj_sorted
    return q_out


def compute_partial_r2(
    y_true: np.ndarray,
    pred_reduced: np.ndarray,
    pred_full: np.ndarray,
) -> float:
    """Exact out-of-fold Partial R^2 of `pred_full` beyond `pred_reduced`:
        Partial R^2 = (SSE_reduced - SSE_full) / SSE_reduced
                    = (R^2_full - R^2_reduced) / (1 - R^2_reduced).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    pr = np.asarray(pred_reduced, dtype=np.float64)
    pf = np.asarray(pred_full, dtype=np.float64)
    sse_red = float(np.sum((yt - pr) ** 2))
    sse_full = float(np.sum((yt - pf) ** 2))
    if sse_red <= 1e-12:
        return 0.0
    return float((sse_red - sse_full) / sse_red)


def bootstrap_ci_and_lift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute exact 4-decimal OOF R^2, Spearman rho, RMSE, 2,000x Bootstrap 95% CIs,
    paired Delta R^2, Partial R^2 (+ 95% CI), and calibrated one-sided bootstrap p-value."""
    import warnings

    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yb_arr = np.asarray(baseline_pred, dtype=np.float64) if baseline_pred is not None else None
    valid = np.isfinite(yt) & np.isfinite(yp)
    if yb_arr is not None:
        valid = valid & np.isfinite(yb_arr)
    yt = yt[valid]
    yp = yp[valid]
    if yb_arr is not None:
        yb_arr = yb_arr[valid]
    n = len(yt)
    if n < 2:
        raise ValueError(f"bootstrap_ci_and_lift requires at least 2 finite samples, got {n}")
    rng = np.random.default_rng(seed)

    r2 = float(r2_score(yt, yp)) if float(np.var(yt)) > 1e-12 else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, rho_p = stats.spearmanr(yt, yp)
    rho = float(rho) if np.isfinite(rho) else 0.0
    rmse = float(root_mean_squared_error(yt, yp))

    boot_r2 = np.empty(n_boot, dtype=np.float64)
    boot_rho = np.empty(n_boot, dtype=np.float64)
    boot_dr2 = np.empty(n_boot, dtype=np.float64) if yb_arr is not None else None
    boot_pr2 = np.empty(n_boot, dtype=np.float64) if yb_arr is not None else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt_b = yt[idx]
            yp_b = yp[idx]
            has_var = float(np.var(yt_b)) > 1e-12
            boot_r2[b] = r2_score(yt_b, yp_b) if has_var else 0.0
            rb = stats.spearmanr(yt_b, yp_b)[0]
            boot_rho[b] = rb if np.isfinite(rb) else 0.0
            if yb_arr is not None and boot_dr2 is not None and boot_pr2 is not None:
                yb_b = yb_arr[idx]
                base_b = r2_score(yt_b, yb_b) if has_var else 0.0
                boot_dr2[b] = boot_r2[b] - base_b
                boot_pr2[b] = compute_partial_r2(yt_b, yb_b, yp_b)

    out: dict[str, Any] = {
        "n": int(n),
        "cv_r2": round(r2, 4),
        "cv_r2_exact": r2,
        "cv_r2_ci95": [round(float(np.nanpercentile(boot_r2, 2.5)), 4), round(float(np.nanpercentile(boot_r2, 97.5)), 4)],
        "spearman_rho": round(rho, 4),
        "spearman_p": float(rho_p) if np.isfinite(rho_p) else 1.0,
        "rho_ci95": [round(float(np.nanpercentile(boot_rho, 2.5)), 4), round(float(np.nanpercentile(boot_rho, 97.5)), 4)],
        "rmse": round(rmse, 4),
        "n_boot": int(n_boot),
    }

    if yb_arr is not None and boot_dr2 is not None and boot_pr2 is not None:
        base_r2 = float(r2_score(yt, yb_arr)) if float(np.var(yt)) > 1e-12 else 0.0
        delta_r2 = r2 - base_r2
        part_r2 = compute_partial_r2(yt, yb_arr, yp)
        # Phipson-Smyth (+1)/(B+1) calibrated one-sided bootstrap p-value (strictly > 0 for BH-FDR)
        p_one = float((np.sum(boot_dr2 <= 0.0) + 1.0) / (n_boot + 1.0))
        out.update(
            {
                "baseline_r2": round(base_r2, 4),
                "delta_r2": round(delta_r2, 4),
                "delta_r2_ci95": [
                    round(float(np.nanpercentile(boot_dr2, 2.5)), 4),
                    round(float(np.nanpercentile(boot_dr2, 97.5)), 4),
                ],
                "partial_r2": round(part_r2, 4),
                "partial_r2_ci95": [
                    round(float(np.nanpercentile(boot_pr2, 2.5)), 4),
                    round(float(np.nanpercentile(boot_pr2, 97.5)), 4),
                ],
                "p_one_sided": round(p_one, 4),
                "p_one_sided_exact": p_one,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Zero-Leakage Player & Draft-Year Group Splits
# ---------------------------------------------------------------------------

def build_zero_leakage_group_splits(
    nfl_ids: np.ndarray,
    draft_years: np.ndarray,
    combine_positions: np.ndarray | list[str],
    mode: str = "stratified_player",
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Build strictly leak-free player-disjoint or draft-year-disjoint CV splits.

    - `stratified_player`: Exact 5-fold StratifiedKFold on unique players balanced across
      `combine_position` (matching V2 outer splits bit-for-bit when sorted by `nfl_id`).
    - `temporal_loyo`: Strict GroupKFold grouped by `draft_year` (Leave-One-Draft-Year-Out),
      guaranteeing zero player overlap AND zero draft-year overlap.
    - `player_group`: GroupKFold grouped strictly on `nfl_ids` (handles multi-row tables).
    """
    ids = np.asarray(nfl_ids)
    yrs = np.asarray(draft_years)
    pos = np.asarray(combine_positions)
    n = len(ids)

    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    if mode == "stratified_player":
        if len(np.unique(ids)) != n:
            raise ValueError("stratified_player expects 1 row per unique nfl_id; use player_group for multi-row data.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (tr, va) in enumerate(skf.split(np.zeros(n), pos)):
            splits.append((tr, va, f"fold_{i + 1}"))
    elif mode == "temporal_loyo":
        uniq_yrs = sorted(np.unique(yrs).tolist())
        gkf = GroupKFold(n_splits=len(uniq_yrs))
        raw_splits = list(gkf.split(np.zeros(n), groups=yrs))
        # Sort deterministically by held-out draft year
        raw_splits.sort(key=lambda pair: int(yrs[pair[1][0]]))
        for tr, va in raw_splits:
            held_yr = int(yrs[va[0]])
            splits.append((tr, va, f"holdout_{held_yr}"))
    elif mode == "player_group":
        gkf = GroupKFold(n_splits=n_splits)
        for i, (tr, va) in enumerate(gkf.split(np.zeros(n), groups=ids)):
            splits.append((tr, va, f"group_fold_{i + 1}"))
    else:
        raise ValueError(f"Unknown CV mode '{mode}'")

    # Runtime zero-leakage invariant enforcement
    seen_va: list[int] = []
    for tr, va, fname in splits:
        if len(set(ids[tr].tolist()) & set(ids[va].tolist())) > 0:
            raise AssertionError(f"Zero-leakage violation in {fname}: player ID appears in both train and val!")
        if mode == "temporal_loyo":
            if len(set(yrs[tr].tolist()) & set(yrs[va].tolist())) > 0:
                raise AssertionError(f"Temporal leakage violation in {fname}: draft_year overlaps train and val!")
        seen_va.extend(va.tolist())
    if sorted(seen_va) != list(range(n)):
        raise AssertionError("CV splits do not partition all samples exactly once.")
    return splits


# ---------------------------------------------------------------------------
# Zero-Leakage Positional Residual Ridge / GBDT Meta-Fusion
# ---------------------------------------------------------------------------

def _fit_nonneg_ridge_weights(X_inner: np.ndarray, y: np.ndarray, l2_reg: float = 0.06) -> tuple[np.ndarray, float]:
    """Constrained non-negative L2 meta-blender regularized smoothly toward equal-weight prior."""
    k = X_inner.shape[1]
    if k == 1:
        return np.array([1.0], dtype=np.float64), 0.0

    def _obj(params: np.ndarray) -> float:
        w = params[:k]
        b = params[k]
        pred = X_inner @ w + b
        prior = np.ones(k) / float(k)
        return float(np.mean((y - pred) ** 2) + l2_reg * np.sum((w - prior) ** 2))

    x0 = np.concatenate([np.ones(k) / float(k), [0.0]])
    bounds = [(0.0, 1.4)] * k + [(-0.5, 0.5)]
    res = optimize.minimize(_obj, x0, method="L-BFGS-B", bounds=bounds)
    return res.x[:k], float(res.x[k])


@dataclass
class PositionalResidualMetaFusion:
    """Strictly within-fold residual meta-fusion combining:
    1. Global optical physics tabular OOF (Ridge + LightGBM + XGBoost)
    2. V2 unified 1D-CNN OOF (global sequence prior)
    3. V4 position-specific multi-scale 1D-CNN + Dual-Head Attention OOF + attention-weighted physics
    4. Smooth Empirical-Bayes positional meta-weights + nested inner-CV regularized residual RidgeCV.
    """

    l2_reg: float = 0.06
    residual_blend: float | None = None

    def fit_predict_fold(
        self,
        tr_blocks_inner: np.ndarray,   # (N_tr, M) inner-CV OOF modality predictions
        va_blocks: np.ndarray,         # (N_va, M) outer-fold modality predictions
        tr_pos_extra: np.ndarray,      # (N_tr, P) V4 positional neural + attention-physics features
        va_pos_extra: np.ndarray,      # (N_va, P)
        tr_groups: np.ndarray,         # (N_tr,) position group strings
        va_groups: np.ndarray,         # (N_va,)
        y_tr: np.ndarray,              # (N_tr,)
    ) -> np.ndarray:
        """Fit meta-weights and nested-CV position-specific residual correctors strictly on training fold."""
        from sklearn.model_selection import KFold

        X_in = np.nan_to_num(tr_blocks_inner.astype(np.float64), nan=0.0)
        X_va = np.nan_to_num(va_blocks.astype(np.float64), nan=0.0)
        y = np.nan_to_num(np.asarray(y_tr, dtype=np.float64), nan=0.0)

        # 1. Global constrained non-negative meta-blend (strictly fit on inner OOFs)
        w_glob, b_glob = _fit_nonneg_ridge_weights(X_in, y, l2_reg=self.l2_reg)
        va_base = X_va @ w_glob + b_glob

        # 2. Smooth Empirical-Bayes group meta-weights + regularized residual RidgeCV blend
        Z_tr = np.nan_to_num(tr_pos_extra.astype(np.float64), nan=0.0)
        Z_va = np.nan_to_num(va_pos_extra.astype(np.float64), nan=0.0)

        va_out = va_base.copy()
        alphas = np.logspace(0.5, 3.8, 18)
        if self.residual_blend is not None and float(self.residual_blend) < 0.0:
            raise ValueError(f"residual_blend must be >= 0.0, got {self.residual_blend}")
        cand_blends = [0.0, 0.20, 0.40] if self.residual_blend is None else [float(self.residual_blend)]

        for grp in POSITION_GROUPS:
            g_tr = (tr_groups == grp)
            g_va = (va_groups == grp)
            n_g = int(g_tr.sum())
            if int(g_va.sum()) == 0 or n_g < 20:
                continue

            # Standardize within position group strictly on training fold
            mu = Z_tr[g_tr].mean(axis=0, keepdims=True)
            sd = np.maximum(Z_tr[g_tr].std(axis=0, keepdims=True), 1e-4)
            z_tr_s = np.clip((Z_tr[g_tr] - mu) / sd, -4.0, 4.0)
            z_va_s = np.clip((Z_va[g_va] - mu) / sd, -4.0, 4.0)

            # Analytical Empirical-Bayes shrinkage of group meta-weights toward global weights
            w_grp, b_grp = _fit_nonneg_ridge_weights(X_in[g_tr], y[g_tr], l2_reg=0.25)
            shrink_g = float(n_g) / (float(n_g) + 90.0)
            w_eb = shrink_g * w_grp + (1.0 - shrink_g) * w_glob
            b_eb = shrink_g * b_grp + (1.0 - shrink_g) * b_glob
            tr_eb_g = X_in[g_tr] @ w_eb + b_eb
            va_eb_g = X_va[g_va] @ w_eb + b_eb
            res_g = y[g_tr] - tr_eb_g

            if len(cand_blends) == 1:
                bl_opt = cand_blends[0]
            else:
                # Strictly nested 5-fold inner CV on training group to select 1D residual blend strength
                kf = KFold(n_splits=min(5, max(2, n_g // 10)), shuffle=True, random_state=42)
                in_res_pred = np.zeros(n_g, dtype=np.float64)
                for itr, iva in kf.split(z_tr_s):
                    in_res_pred[iva] = RidgeCV(alphas=alphas).fit(z_tr_s[itr], res_g[itr]).predict(z_tr_s[iva])

                best_mse = float("inf")
                bl_opt = 0.0
                for bl in cand_blends:
                    mse = float(np.mean((res_g - bl * in_res_pred) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        bl_opt = bl

            if bl_opt > 0.0:
                ridge_res = RidgeCV(alphas=alphas).fit(z_tr_s, res_g)
                va_out[g_va] = va_eb_g + bl_opt * ridge_res.predict(z_va_s)
            else:
                va_out[g_va] = va_eb_g

        return np.nan_to_num(va_out, nan=0.0)
