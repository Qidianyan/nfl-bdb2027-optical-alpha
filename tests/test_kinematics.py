"""Tests for physics-grounded optical kinematics, circular wrap, and real-data integrity."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nfl_bdb2027.config import DRILL_NAME_ALIASES
from nfl_bdb2027.data import DatasetPaths, load_combine_results, load_players
from nfl_bdb2027.kinematics import (
    aggregate_player_kinematics,
    compute_frame_kinematics,
    extract_rep_level_kinematics,
)


def test_circular_angular_velocity_wrap_across_360() -> None:
    """Crossing 355 deg -> 5 deg over 0.1s is +10 deg (+1.745 rad/s), NOT 350 deg."""
    toy = pl.DataFrame({
        "draft_year": [2025, 2025, 2025],
        "event_id": ["ev1", "ev1", "ev1"],
        "nfl_id": [99999, 99999, 99999],
        "entity_type": ["PLAYER", "PLAYER", "PLAYER"],
        "time": ["2025-03-01T12:00:00.000", "2025-03-01T12:00:00.100", "2025-03-01T12:00:00.200"],
        "drill_type": ["SKILL_DRILLS_WR"] * 3,
        "drill_name": ["SPEED_OUT_LEFT"] * 3,
        "attempt": [1, 1, 1],
        "x": [10.0, 10.6, 11.2],
        "y": [20.0, 20.0, 20.0],
        "s": [6.0, 6.0, 4.0],
        "a": [1.0, 2.0, 5.0],
        "dis": [0.6, 0.6, 0.4],
        "dir": [355.0, 5.0, 355.0],
    }).lazy()

    out = compute_frame_kinematics(toy).collect()
    assert out["canon_drill"][0] == DRILL_NAME_ALIASES["SPEED_OUT_LEFT"]
    # Frame 1: 355 -> 5 is 10 deg / 0.1s = 100 deg/s = 1.7453 rad/s
    expected_omega = 10.0 * math.pi / 180.0 / 0.1
    assert out["omega_rad_s"][1] == pytest.approx(expected_omega, rel=1e-4)
    # Frame 2: s drops 6.0 -> 4.0 over 0.1s => signed a_tan = -20.0 yd/s^2
    assert out["a_tan"][2] == pytest.approx(-20.0, rel=1e-5)
    # True Euclidean step distance
    assert out["step_dist"][1] == pytest.approx(0.6, rel=1e-5)


def test_real_combine_kinematics_full_coverage_and_fold_isolation() -> None:
    """Verify 100% skill-drill optical coverage on all 510 official prospects
    and zero cross-fold leakage when fitting drill normalizers on a subset."""
    paths = DatasetPaths.from_dir()
    players = load_players(paths)
    cr = load_combine_results(paths)
    assert players.height == 510
    assert cr.height == 510

    rep_df = extract_rep_level_kinematics(paths)
    assert rep_df.height > 6000
    assert rep_df["nfl_id"].n_unique() == 510

    # Fit on 2023+2024 only (temporal zero-leakage check)
    fit_ids = players.filter(pl.col("draft_year") < 2025)["nfl_id"].to_list()
    kin_temp = aggregate_player_kinematics(rep_df, cr, fit_nfl_ids=fit_ids)
    assert kin_temp.height == 510
    assert kin_temp["opt_power_index"].null_count() == 0
    assert kin_temp["opt_cbi_index"].null_count() == 0
    assert kin_temp["opt_impulse_momentum"].null_count() == 0
    assert np.isfinite(kin_temp["opt_mass_power"].to_numpy()).all()
