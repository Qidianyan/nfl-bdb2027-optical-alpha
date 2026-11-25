"""Physics-grounded, 100% Polars vectorized optical Combine kinematic feature engineering."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from .config import DRILL_NAME_ALIASES, DT_SEC, LBS_TO_KG
from .data import DatasetPaths, scan_combine_tracking

REP_GROUP_COLS = ["nfl_id", "draft_year", "drill_type", "canon_drill", "event_id", "attempt"]

SKILL_Z_METRICS = [
    "max_s",
    "p95_s",
    "max_a",
    "p95_a",
    "p95_atan_pos",
    "p95_decel",
    "p95_jerk",
    "p95_omega",
    "p95_acent",
    "p90_kappa_fast",
    "max_power",
    "p95_power",
    "p95_brake_power",
    "p95_accel_power",
    "reaccel_out_of_break",
    "first1s_a",
    "turn_apex_s",
    "vel_ret",
    "path_eff",
]


def compute_frame_kinematics(ct_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute signed tangential acceleration, jerk, circular angular velocity,
    path curvature, centripetal acceleration, and mechanical power at 10 Hz.

    Notes on real-data sensor physics:
    1. Raw `a` in combine_tracking.csv is unsigned acceleration magnitude (>= 0).
       We derive signed tangential acceleration `a_tan = ds / dt` to distinguish
       forward propulsion from deceleration/braking.
    2. Raw `dis` exhibits a low-speed smoothing attenuation relative to Euclidean
       `(x, y)` displacement. We compute true step distance `sqrt(dx^2 + dy^2)`
       directly from optical coordinates for accurate path efficiency.
    3. Cross-year drill naming discrepancies (e.g., LINE vs LINE_DRILL) are
       canonicalized via DRILL_NAME_ALIASES before aggregation.
    """
    return (
        ct_lf.filter(pl.col("entity_type") == "PLAYER")
        .with_columns([
            pl.col("time").str.to_datetime(strict=False).alias("ts"),
            pl.col("drill_name").replace(DRILL_NAME_ALIASES).alias("canon_drill"),
        ])
        .sort(REP_GROUP_COLS + ["ts"])
        .with_columns([
            pl.col("x").diff().over(REP_GROUP_COLS).alias("dx"),
            pl.col("y").diff().over(REP_GROUP_COLS).alias("dy"),
            pl.col("s").diff().over(REP_GROUP_COLS).alias("ds"),
            pl.col("a").diff().over(REP_GROUP_COLS).alias("da"),
            pl.col("dir").diff().over(REP_GROUP_COLS).alias("raw_ddir"),
            pl.arange(0, pl.len()).over(REP_GROUP_COLS).alias("frame_idx"),
            pl.len().over(REP_GROUP_COLS).alias("n_frames"),
        ])
        .with_columns([
            (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().fill_null(0.0).alias("step_dist"),
            (pl.col("ds") / DT_SEC).alias("a_tan"),
            (pl.col("da") / DT_SEC).alias("jerk"),
            # Shortest circular angular displacement in degrees: [0, 180]
            (((pl.col("raw_ddir") + 180.0) % 360.0) - 180.0).abs().alias("abs_ddir_deg"),
        ])
        .with_columns([
            (pl.col("abs_ddir_deg") * (math.pi / 180.0) / DT_SEC).alias("omega_rad_s"),
        ])
        .with_columns([
            # Centripetal lateral acceleration: a_c = v * omega = v^2 * kappa
            (pl.col("s") * pl.col("omega_rad_s")).alias("a_cent"),
            # Path curvature kappa = omega / v (defined when moving > 1.5 yd/s)
            pl.when(pl.col("s") > 1.5)
            .then(pl.col("omega_rad_s") / pl.col("s"))
            .otherwise(0.0)
            .alias("kappa"),
            # Mechanical power proxies (velocity * acceleration, yd^2/s^3)
            (pl.col("s") * pl.col("a")).alias("mech_power"),
            (pl.col("s") * pl.col("a_tan").clip(lower_bound=0.0)).alias("accel_power"),
            (pl.col("s") * (-pl.col("a_tan")).clip(lower_bound=0.0)).alias("brake_power"),
            # Out-of-break re-acceleration within next 0.3s (3 frames)
            pl.col("a_tan").shift(-1).over(REP_GROUP_COLS).alias("_next_atan_1"),
            pl.col("a_tan").shift(-2).over(REP_GROUP_COLS).alias("_next_atan_2"),
            pl.col("a_tan").shift(-3).over(REP_GROUP_COLS).alias("_next_atan_3"),
        ])
        .with_columns([
            pl.max_horizontal("_next_atan_1", "_next_atan_2", "_next_atan_3").alias("reaccel_03s"),
        ])
        .drop(["_next_atan_1", "_next_atan_2", "_next_atan_3"])
    )


def extract_rep_level_kinematics(paths: DatasetPaths) -> pl.DataFrame:
    """Aggregate 10 Hz optical tracking into per-drill-rep biomechanical summaries."""
    ct_k = compute_frame_kinematics(scan_combine_tracking(paths))

    rep_df = (
        ct_k.group_by(REP_GROUP_COLS)
        .agg([
            pl.len().alias("n_frames"),
            pl.col("s").max().alias("max_s"),
            pl.col("s").quantile(0.95).alias("p95_s"),
            pl.col("s").mean().alias("mean_s"),
            pl.col("a").max().alias("max_a"),
            pl.col("a").quantile(0.95).alias("p95_a"),
            pl.col("a_tan").quantile(0.95).alias("p95_atan_pos"),
            (-pl.col("a_tan").quantile(0.05)).alias("p95_decel"),
            pl.col("jerk").abs().quantile(0.95).alias("p95_jerk"),
            pl.col("jerk").max().alias("max_pos_jerk"),
            pl.col("omega_rad_s").quantile(0.95).alias("p95_omega"),
            pl.col("a_cent").quantile(0.95).alias("p95_acent"),
            pl.col("a_cent").max().alias("max_acent"),
            pl.col("kappa").filter(pl.col("s") > 2.5).quantile(0.90).alias("p90_kappa_fast"),
            pl.col("mech_power").max().alias("max_power"),
            pl.col("mech_power").quantile(0.95).alias("p95_power"),
            pl.col("brake_power").quantile(0.95).alias("p95_brake_power"),
            pl.col("accel_power").quantile(0.95).alias("p95_accel_power"),
            pl.col("reaccel_03s")
            .filter(pl.col("a_tan") < -1.5)
            .quantile(0.90)
            .alias("reaccel_out_of_break"),
            # First 1.0s (frames 0..9) explosive launch signature
            pl.col("a").filter(pl.col("frame_idx") < 10).mean().alias("first1s_a"),
            pl.col("s").filter(pl.col("frame_idx") == 10).first().alias("s_at_1s"),
            # Velocity retention in final 25% of drill rep
            pl.col("s")
            .filter(pl.col("frame_idx") >= (pl.col("n_frames") * 0.75))
            .mean()
            .alias("late_s"),
            # Minimum/mean speed maintained during sharp turning (>1.5 rad/s)
            pl.col("s").filter(pl.col("omega_rad_s") > 1.5).mean().alias("turn_apex_s"),
            # True Euclidean path length & net displacement
            pl.col("step_dist").sum().alias("path_len"),
            (
                (pl.col("x").last() - pl.col("x").first()).pow(2)
                + (pl.col("y").last() - pl.col("y").first()).pow(2)
            )
            .sqrt()
            .alias("net_disp"),
            (pl.col("y").max() - pl.col("y").min()).alias("y_span"),
        ])
        .with_columns([
            (pl.col("late_s") / pl.col("max_s").clip(lower_bound=0.5)).alias("vel_ret"),
            (pl.col("net_disp") / pl.col("path_len").clip(lower_bound=0.5))
            .clip(upper_bound=1.0)
            .alias("path_eff"),
        ])
        .collect()
    )
    return rep_df


def aggregate_player_kinematics(
    rep_df: pl.DataFrame,
    combine_results: pl.DataFrame,
    fit_nfl_ids: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Aggregate rep-level kinematics to player-level signatures.

    If `fit_nfl_ids` is provided, within-drill z-score normalization parameters
    (mean, std per canonical drill) are estimated STRICTLY from `fit_nfl_ids`
    to guarantee zero covariate leakage under cross-validation or temporal splits.
    """
    sk_all = rep_df.filter(pl.col("drill_type").str.starts_with("SKILL_DRILLS_"))
    sk_fit = (
        sk_all.filter(pl.col("nfl_id").is_in(list(fit_nfl_ids)))
        if fit_nfl_ids is not None
        else sk_all
    )

    # Fit per-drill mean & std strictly on fit split, fallback to global skill stats
    global_stats = sk_fit.select(
        [pl.col(c).mean().alias(f"__gmean_{c}") for c in SKILL_Z_METRICS]
        + [pl.col(c).std().alias(f"__gstd_{c}") for c in SKILL_Z_METRICS]
    ).to_dicts()[0]

    drill_stats = sk_fit.group_by("canon_drill").agg(
        [pl.col(c).mean().alias(f"__dmean_{c}") for c in SKILL_Z_METRICS]
        + [pl.col(c).std().alias(f"__dstd_{c}") for c in SKILL_Z_METRICS]
        + [pl.len().alias("__dn")]
    )

    sk_joined = sk_all.join(drill_stats, on="canon_drill", how="left")
    z_exprs = []
    for c in SKILL_Z_METRICS:
        gm = float(global_stats[f"__gmean_{c}"] or 0.0)
        gs = float(global_stats[f"__gstd_{c}"] or 1.0)
        if gs < 1e-6:
            gs = 1.0
        # Empirical-Bayes shrink rare drill stats toward global skill distribution (k=8)
        d_n = pl.col("__dn").fill_null(0.0).cast(pl.Float64)
        d_m = pl.col(f"__dmean_{c}").fill_null(gm)
        d_s = pl.col(f"__dstd_{c}").fill_null(gs)
        shrunk_m = (d_n * d_m + 8.0 * gm) / (d_n + 8.0)
        shrunk_s = ((d_n * d_s + 8.0 * gs) / (d_n + 8.0)).clip(lower_bound=1e-4)
        z_exprs.append(((pl.col(c) - shrunk_m) / shrunk_s).clip(-4.0, 4.0).alias(f"z_{c}"))

    sk_z = sk_joined.with_columns(z_exprs)

    # Attempt fatigue decay: change in z_p95_power from earliest to latest rep within multi-rep drills
    fatigue_df = (
        sk_z.sort(["nfl_id", "canon_drill", "attempt"])
        .group_by(["nfl_id", "canon_drill"])
        .agg([
            pl.col("z_p95_power").first().alias("first_pow"),
            pl.col("z_p95_power").last().alias("last_pow"),
            pl.len().alias("n_att"),
        ])
        .filter(pl.col("n_att") >= 2)
        .group_by("nfl_id")
        .agg((pl.col("last_pow") - pl.col("first_pow")).mean().alias("opt_fatigue_decay"))
    )

    skill_player = sk_z.group_by("nfl_id").agg([
        # Raw universal optical ceilings
        pl.col("max_s").max().alias("opt_sk_max_s"),
        pl.col("max_a").max().alias("opt_sk_max_a"),
        pl.col("p95_power").mean().alias("opt_sk_raw_power"),
        pl.col("p95_decel").mean().alias("opt_sk_raw_decel"),
        pl.col("p95_acent").mean().alias("opt_sk_raw_acent"),
        pl.col("p95_omega").mean().alias("opt_sk_raw_omega"),
        pl.col("p95_jerk").mean().alias("opt_sk_raw_jerk"),
        # Within-drill standardized optical composites (100% coverage across 510 players)
        pl.col("z_p95_power").mean().alias("opt_power_index"),
        pl.col("z_max_s").mean().alias("opt_speed_index"),
        (0.60 * pl.col("z_p95_a").mean() + 0.40 * pl.col("z_p95_jerk").mean()).alias("opt_burst_index"),
        (
            0.35 * pl.col("z_p95_acent").mean()
            + 0.30 * pl.col("z_p95_decel").mean()
            + 0.20 * pl.col("z_p95_omega").mean()
            + 0.15 * pl.col("z_reaccel_out_of_break").mean()
        ).alias("opt_cbi_index"),
        pl.col("z_p95_omega").mean().alias("opt_omega_z"),
        pl.col("z_p95_acent").mean().alias("opt_acent_z"),
        pl.col("z_p95_decel").mean().alias("opt_decel_z"),
        pl.col("z_p95_jerk").mean().alias("opt_jerk_z"),
        pl.col("z_reaccel_out_of_break").mean().alias("opt_reaccel_z"),
        pl.col("z_turn_apex_s").mean().alias("opt_turn_apex_z"),
        pl.col("z_first1s_a").mean().alias("opt_first1s_a_z"),
        pl.col("z_vel_ret").mean().alias("opt_vel_ret_z"),
        pl.col("z_path_eff").mean().alias("opt_path_eff_z"),
        pl.col("z_p95_power").max().alias("opt_peak_rep_power_z"),
        # Consistency across drill reps (lower = tighter mechanical repeatability)
        pl.col("z_p95_power").std().alias("opt_rep_consistency_sd"),
        pl.len().alias("opt_n_skill_reps"),
    ])

    forty_player = (
        rep_df.filter(pl.col("drill_type") == "FORTY_YARD_DASH")
        .group_by("nfl_id")
        .agg([
            pl.col("max_s").max().alias("opt_f40_max_s"),
            pl.col("max_a").max().alias("opt_f40_max_a"),
            pl.col("first1s_a").max().alias("opt_f40_first1s_a"),
            pl.col("s_at_1s").max().alias("opt_f40_s_at_1s"),
            pl.col("p95_jerk").max().alias("opt_f40_jerk"),
            pl.col("vel_ret").max().alias("opt_f40_vel_ret"),
            pl.col("path_eff").max().alias("opt_f40_path_eff"),
            pl.col("y_span").min().alias("opt_f40_lat_drift"),
            pl.col("max_power").max().alias("opt_f40_max_power"),
            pl.col("max_s").std().alias("opt_f40_attempt_sd"),
        ])
    )

    cod_player = (
        rep_df.filter(pl.col("drill_type").is_in(["THREE_CONE_DRILL", "SHORT_SHUTTLE"]))
        .group_by("nfl_id")
        .agg([
            pl.col("p95_acent").max().alias("opt_cod_p95_acent"),
            pl.col("p95_decel").max().alias("opt_cod_p95_decel"),
            pl.col("turn_apex_s").max().alias("opt_cod_turn_apex_s"),
            pl.col("reaccel_out_of_break").max().alias("opt_cod_reaccel"),
            pl.col("p95_power").max().alias("opt_cod_p95_power"),
        ])
    )

    out = (
        combine_results.select(["nfl_id", "combine_weight"])
        .join(skill_player, on="nfl_id", how="left")
        .join(fatigue_df, on="nfl_id", how="left")
        .join(forty_player, on="nfl_id", how="left")
        .join(cod_player, on="nfl_id", how="left")
        .with_columns((pl.col("combine_weight").cast(pl.Float64) * LBS_TO_KG).alias("mass_kg"))
        .with_columns([
            # Momentum impulse & mass-scaled mechanical power (force = m*a, power = m*a*v)
            (pl.col("mass_kg") * pl.col("opt_sk_max_a")).alias("opt_impulse_momentum"),
            (pl.col("mass_kg") * pl.col("opt_sk_raw_power")).alias("opt_mass_power"),
            (pl.col("mass_kg") * pl.col("opt_sk_raw_decel")).alias("opt_braking_force"),
            (pl.col("mass_kg") * pl.col("opt_f40_max_a")).alias("opt_f40_impulse"),
        ])
        .drop(["combine_weight", "mass_kg"])
    )
    return out


def extract_player_drill_trajectories(
    paths: DatasetPaths,
    nfl_ids: Sequence[int],
    drill_type: str | None = None,
    canon_drill: str | None = None,
) -> pd.DataFrame:
    """Extract real 10 Hz frame-by-frame trajectories for selected players & drills
    to power publication visualizations (Figure 2)."""
    lf = compute_frame_kinematics(scan_combine_tracking(paths)).filter(
        pl.col("nfl_id").is_in(list(nfl_ids))
    )
    if drill_type is not None:
        lf = lf.filter(pl.col("drill_type") == drill_type)
    if canon_drill is not None:
        lf = lf.filter(pl.col("canon_drill") == canon_drill)

    df = (
        lf.select([
            "nfl_id",
            "drill_type",
            "canon_drill",
            "event_id",
            "attempt",
            "frame_idx",
            (pl.col("frame_idx").cast(pl.Float64) * DT_SEC).alias("t_sec"),
            "x",
            "y",
            "s",
            "a",
            "a_tan",
            "a_cent",
            "omega_rad_s",
            "mech_power",
        ])
        .collect()
        .to_pandas()
    )
    return df
