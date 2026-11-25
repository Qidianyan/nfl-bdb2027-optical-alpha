"""Real NFL regular-season game outcome aggregation from 22.65M tracking frames,
314k play-level records, and 510 career success records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold

from .config import LATEST_SEASON_PLUS_ONE, UDFA_IMPUTED_PICK
from .data import (
    DatasetPaths,
    load_career_successes,
    load_combine_results,
    load_players,
    scan_game_tracking,
    scan_player_play,
)


def extract_game_tracking_outcomes(paths: DatasetPaths) -> pl.DataFrame:
    """Lazily aggregate all game_tracking_*.csv files (22.65M+ frames) per player."""
    gt_lf = scan_game_tracking(paths)
    return (
        gt_lf.group_by("nfl_id")
        .agg([
            pl.len().alias("gt_frames"),
            pl.struct(["game_id", "play_id"]).n_unique().alias("gt_plays"),
            pl.col("s").max().alias("game_max_s"),
            pl.col("s").quantile(0.99).alias("game_p99_s"),
            pl.col("s").quantile(0.95).alias("game_p95_s"),
            pl.col("a").abs().quantile(0.95).alias("game_p95_abs_a"),
            pl.col("a").quantile(0.99).alias("game_p99_pos_a"),
            (-pl.col("a").quantile(0.01)).alias("game_p99_decel"),
            (pl.col("s") * pl.col("a").abs()).quantile(0.95).alias("game_p95_power"),
            (pl.col("s") >= 7.0).mean().alias("game_high_speed_rate"),
        ])
        .collect()
    )


def extract_player_play_outcomes(paths: DatasetPaths) -> pl.DataFrame:
    """Aggregate play-level on-field NFL performance from player_play.csv."""
    pp_lf = scan_player_play(paths)
    return (
        pp_lf.group_by("nfl_id")
        .agg([
            pl.len().alias("pp_snaps"),
            pl.col("expected_points_added").mean().alias("pp_epa_mean"),
            pl.col("separation_at_pass_forward").mean().alias("pp_sep_mean"),
            pl.col("separation_at_pass_forward").count().alias("pp_sep_n"),
            pl.col("cushion").mean().alias("pp_cushion_mean"),
            pl.col("cushion").count().alias("pp_cushion_n"),
            (pl.col("yards_after_catch") - pl.col("expected_yards_after_catch"))
            .mean()
            .alias("pp_yac_oe_mean"),
            (pl.col("yards_after_catch") - pl.col("expected_yards_after_catch"))
            .count()
            .alias("pp_yac_oe_n"),
            pl.col("target").sum().alias("pp_targets"),
            pl.col("rec_yards").sum().alias("pp_rec_yards"),
            pl.col("player_get_off").mean().alias("pp_get_off_mean"),
            pl.col("player_get_off").count().alias("pp_get_off_n"),
            pl.col("time_to_pressure").mean().alias("pp_ttp_mean"),
            pl.col("time_to_pressure").count().alias("pp_ttp_n"),
            pl.col("time_to_qb_hurry").mean().alias("pp_tth_mean"),
            pl.col("quick_pressure").mean().alias("pp_quick_press_rate"),
            pl.col("peak_pressure_probability_allowed").mean().alias("pp_ol_press_prob_mean"),
            pl.col("peak_pressure_probability_allowed").count().alias("pp_ol_press_n"),
            pl.col("pressure_allowed").mean().alias("pp_ol_press_rate"),
            pl.col("tackle").sum().alias("pp_tackles"),
            pl.col("sack").sum().alias("pp_sacks"),
        ])
        .with_columns([
            (pl.col("pp_tackles").cast(pl.Float64) / pl.col("pp_snaps").clip(lower_bound=1)).alias(
                "pp_tackle_rate"
            ),
        ])
        .collect()
    )


def _pos_zscore(series: pd.Series, pos: pd.Series) -> pd.Series:
    return series.groupby(pos).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def build_outcomes_table(paths: DatasetPaths) -> pd.DataFrame:
    """Build unified 510-prospect table linking metadata, combine results,
    exposure-normalized career successes, play-level outcomes, game tracking,
    and Draft-Capital Residual Alpha."""
    players = load_players(paths)
    cr = load_combine_results(paths)
    cs = load_career_successes(paths)
    gt = extract_game_tracking_outcomes(paths)
    pp = extract_player_play_outcomes(paths)

    merged = (
        players.join(cr, on=["nfl_id", "draft_year"], how="inner")
        .join(cs, on="nfl_id", how="inner")
        .join(gt, on="nfl_id", how="left")
        .join(pp, on="nfl_id", how="left")
        .with_columns([
            (LATEST_SEASON_PLUS_ONE - pl.col("draft_year")).cast(pl.Float64).alias("eligible_seasons"),
            (pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps"))
            .cast(pl.Float64)
            .alias("scrimmage_snaps"),
            pl.col("draft_overall_pick").cast(pl.Float64).fill_null(UDFA_IMPUTED_PICK).alias("pick_filled"),
            pl.col("draft_overall_pick").is_null().cast(pl.Int8).alias("is_udfa"),
            pl.col("forty").is_null().cast(pl.Int8).alias("opted_out_40"),
            pl.col("three_cone").is_null().cast(pl.Int8).alias("opted_out_3cone"),
            pl.col("short_shuttle").is_null().cast(pl.Int8).alias("opted_out_shuttle"),
        ])
        .with_columns([
            (pl.col("scrimmage_snaps") / pl.col("eligible_seasons")).alias("snaps_per_season"),
            (pl.col("career_games_started").cast(pl.Float64) / pl.col("eligible_seasons")).alias(
                "starts_per_season"
            ),
            (pl.col("career_games_active").cast(pl.Float64) / pl.col("eligible_seasons")).alias(
                "games_active_per_season"
            ),
            (
                (
                    2.0 * pl.col("ap_all_pro_1st_team")
                    + 1.5 * pl.col("ap_all_pro_2nd_team")
                    + 1.0 * pl.col("pro_bowl_original_ballot")
                ).cast(pl.Float64)
                / pl.col("eligible_seasons")
            ).alias("accolade_rate"),
            pl.col("pick_filled").log().alias("log_pick"),
        ])
        .sort("nfl_id")
        .to_pandas()
    )

    pos = merged["combine_position"]

    # 1. Universal Sunday Game Kinematic Power Index (from 22.65M game_tracking frames)
    merged["z_game_p99_s"] = _pos_zscore(merged["game_p99_s"], pos)
    merged["z_game_p95_abs_a"] = _pos_zscore(merged["game_p95_abs_a"], pos)
    merged["z_game_p95_power"] = _pos_zscore(merged["game_p95_power"], pos)
    merged["game_kinematic_index"] = (
        0.35 * merged["z_game_p99_s"]
        + 0.35 * merged["z_game_p95_abs_a"]
        + 0.30 * merged["z_game_p95_power"]
    )

    # 2. Universal Exposure-Normalized NFL Career Value Index (510/510 players)
    merged["z_snaps_per_season"] = _pos_zscore(merged["snaps_per_season"], pos)
    merged["z_starts_per_season"] = _pos_zscore(merged["starts_per_season"], pos)
    merged["z_accolade_rate"] = _pos_zscore(merged["accolade_rate"], pos)
    merged["nfl_value_index"] = (
        0.55 * merged["z_snaps_per_season"]
        + 0.30 * merged["z_starts_per_season"]
        + 0.15 * merged["z_accolade_rate"]
    )

    # 3. Leak-free OOF Expected Value from Draft Capital & Position -> Draft-Capital Residual Alpha
    pos_dum = pd.get_dummies(pos, prefix="pos", dtype=float)
    X_dc = pd.concat([merged[["log_pick"]], pos_dum], axis=1)
    oof_expected = np.zeros(len(merged), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(X_dc, pos):
        reg = RidgeCV(alphas=np.logspace(-2, 3, 25))
        reg.fit(X_dc.iloc[tr_idx], merged["nfl_value_index"].iloc[tr_idx])
        oof_expected[va_idx] = reg.predict(X_dc.iloc[va_idx])
    merged["expected_value_from_draft"] = oof_expected
    merged["draft_capital_alpha"] = merged["nfl_value_index"] - merged["expected_value_from_draft"]

    # 4. Role-specific on-field play dominance z-score (higher = better across every position)
    role_z = pd.Series(np.nan, index=merged.index, dtype=float)
    for p_grp in ["WR", "TE"]:
        m = pos == p_grp
        z_sep = _pos_zscore(merged.loc[m, "pp_sep_mean"], pos.loc[m])
        z_yac = _pos_zscore(merged.loc[m, "pp_yac_oe_mean"], pos.loc[m])
        role_z.loc[m] = pd.concat([z_sep, z_yac], axis=1).mean(axis=1)
    m_dl = pos == "DL"
    z_go = -_pos_zscore(merged.loc[m_dl, "pp_get_off_mean"], pos.loc[m_dl])  # lower get-off is faster
    z_ttp = -_pos_zscore(merged.loc[m_dl, "pp_ttp_mean"], pos.loc[m_dl])  # lower TTP is faster
    role_z.loc[m_dl] = pd.concat([z_go, z_ttp], axis=1).mean(axis=1)
    m_ol = pos == "OL"
    role_z.loc[m_ol] = -_pos_zscore(merged.loc[m_ol, "pp_ol_press_prob_mean"], pos.loc[m_ol])
    m_db = pos == "DB"
    z_cush = -_pos_zscore(merged.loc[m_db, "pp_cushion_mean"], pos.loc[m_db])
    z_tack = _pos_zscore(merged.loc[m_db, "pp_tackle_rate"], pos.loc[m_db])
    role_z.loc[m_db] = pd.concat([z_cush, z_tack], axis=1).mean(axis=1)
    merged["role_onfield_dominance_z"] = role_z

    return merged
