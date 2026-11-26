"""Tests for NFL GM Draft-Room Surplus EV Engine (draft_ev.py) and V5 6-figure / FilmRoom artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nfl_bdb2027.config import FIGURES_DIR, MATERIALS_DIR
from nfl_bdb2027.data import DatasetPaths, load_combine_results
from nfl_bdb2027.draft_ev import (
    DraftSurplusEVEngine,
    pick_to_round,
    realized_production_4yr_millions,
    rookie_contract_4yr_millions,
)
from nfl_bdb2027.kinematics import aggregate_player_kinematics, extract_rep_level_kinematics
from nfl_bdb2027.outcomes import build_outcomes_table
from nfl_bdb2027.positional_attn import POSITION_GROUPS


@pytest.fixture(scope="module")
def fitted_ev_engine() -> DraftSurplusEVEngine:
    paths = DatasetPaths.from_dir()
    rep_df = extract_rep_level_kinematics(paths)
    cr_pl = load_combine_results(paths)
    outcomes_df = build_outcomes_table(paths)
    full_kin = aggregate_player_kinematics(rep_df, cr_pl, fit_nfl_ids=None).to_pandas()
    master_df = outcomes_df.merge(full_kin, on="nfl_id", how="left")
    return DraftSurplusEVEngine.fit(master_df)


def test_cba_rookie_contract_strictly_monotonic_and_bounds() -> None:
    picks = list(range(1, 258))
    costs = [rookie_contract_4yr_millions(p) for p in picks]
    assert costs[0] == pytest.approx(39.49, abs=0.05)
    assert costs[-1] == pytest.approx(3.96, abs=0.05)
    # Strictly non-increasing across every single pick 1..257
    assert all(costs[i] >= costs[i + 1] for i in range(len(costs) - 1))
    assert costs[0] > costs[31] > costs[99] > costs[256]

    assert pick_to_round(1) == 1
    assert pick_to_round(32) == 1
    assert pick_to_round(33) == 2
    assert pick_to_round(177) == 5
    assert pick_to_round(178) == 6
    assert pick_to_round(257) == 7

    for bad in (0, -5, 258, 999, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rookie_contract_4yr_millions(bad)


def test_surplus_ev_conservation_and_optical_monotonicity(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    for grp in POSITION_GROUPS:
        for pk in (10, 45, 115, 177):
            low = fitted_ev_engine.evaluate(grp, pk, -1.5)
            mid = fitted_ev_engine.evaluate(grp, pk, 0.0)
            high = fitted_ev_engine.evaluate(grp, pk, +1.5)

            for res in (low, mid, high):
                # Exact conservation identity: Surplus == Expected Production - Rookie Contract Cost
                assert res["expected_surplus_ev_m"] == pytest.approx(
                    res["expected_production_4yr_m"] - res["rookie_contract_4yr_m"], abs=1e-2
                )
                assert res["surplus_ev_ci95_m"][0] < res["expected_surplus_ev_m"] < res["surplus_ev_ci95_m"][1]
                assert 0.0 <= res["stopwatch_bust_prob"] <= 1.0
                assert 0.0 <= res["v4_optical_bust_prob"] <= 1.0

            # Higher optical alpha strictly increases expected surplus EV and lowers bust probability
            assert high["expected_surplus_ev_m"] > mid["expected_surplus_ev_m"] > low["expected_surplus_ev_m"]
            assert high["v4_optical_bust_prob"] < low["v4_optical_bust_prob"]
            assert high["bust_risk_reduction_pct"] > 0.0


def test_real_historical_comps_retrieval_no_mock_data(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    # Querying WR_TE Day-3 high optical profile must retrieve real cohort prospects from 510 dataset
    valid_ids = set(fitted_ev_engine.prospects_df["nfl_id"].astype(int).tolist())
    for grp in POSITION_GROUPS:
        comps = fitted_ev_engine.find_similar_prospects(grp, draft_pick=140, optical_alpha_z=1.2, top_k=3)
        assert len(comps) == 3
        assert len({c["nfl_id"] for c in comps}) == 3
        for c in comps:
            assert c["nfl_id"] in valid_ids
            assert len(c["display_name"]) >= 3
            assert c["draft_year"] in (2023, 2024, 2025)
            assert 0.0 < c["similarity_score"] <= 1.0


def test_draft_ev_invalid_inputs_intercepted(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    with pytest.raises(ValueError):
        fitted_ev_engine.evaluate("WR_TE", 0, 0.5)
    with pytest.raises(ValueError):
        fitted_ev_engine.evaluate("WR_TE", 260, 0.5)
    with pytest.raises(ValueError):
        fitted_ev_engine.evaluate("WR_TE", 50, float("nan"))
    with pytest.raises(ValueError):
        fitted_ev_engine.evaluate("QUARTERBACK_INVALID", 10, 0.5)


def test_realized_production_non_negative_floor() -> None:
    floor_val = realized_production_4yr_millions(-4.0, 0.0, 0.0, "DB")
    star_val = realized_production_4yr_millions(2.5, 850.0, 16.0, "WR_TE")
    assert floor_val >= 2.4
    assert star_val > floor_val * 5.0


def test_export_ui_bundle_covers_all_510_prospects(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    bundle = fitted_ev_engine.export_ui_and_chart_bundle()
    assert len(bundle["prospects"]) == 510
    assert set(bundle["model_params"].keys()) == set(POSITION_GROUPS)
    assert set(bundle["curves_by_group"].keys()) == set(POSITION_GROUPS)
    for grp in POSITION_GROUPS:
        assert len(bundle["curves_by_group"][grp]["elite_opt_p15"]) >= 12


def test_v5_six_figures_and_interactive_filmroom_integrity() -> None:
    expected_pngs = [
        "fig1_kinematic_transfer_matrix.png",
        "fig2_optical_velocity_decel_profiles.png",
        "fig3_draft_capital_residual_alpha.png",
        "fig4_cv_model_ablation.png",
        "fig5_positional_biomechanical_archetypes.png",
        "fig6_temporal_attention_saliency_and_draft_ev.png",
    ]
    for fname in expected_pngs:
        p = FIGURES_DIR / fname
        assert p.exists(), f"Missing 2x Retina PNG: {p}"
        assert p.stat().st_size > 20_000, f"PNG suspiciously small: {p}"

    filmroom = MATERIALS_DIR / "NFL_BDB2027_Interactive_FilmRoom.html"
    assert filmroom.exists(), f"Missing {filmroom}"
    html = filmroom.read_text(encoding="utf-8")
    assert len(html) > 25_000
    assert "0.2562" in html
    assert "0.2286" in html
    assert "0.6892" in html
    assert "Puka Nacua" in html
    assert 'id="draftPickSlider"' in html
    assert 'id="opticalAlphaSlider"' in html
    assert 'id="saliencyCanvas"' in html
    assert 'id="draftEvCanvas"' in html
    for tok in ("TODO", "FIXME", "PLACEHOLDER"):
        assert tok not in html
