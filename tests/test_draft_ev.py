"""Tests for NFL GM Draft-Room Surplus EV Engine (draft_ev.py) and V5 6-figure / FilmRoom artifacts."""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
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
from test_artifacts import LEGACY_BEIGE_HEXES, _is_forbidden_hue


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
            assert high["v4_optical_bust_prob"] < mid["v4_optical_bust_prob"] < low["v4_optical_bust_prob"]
            assert mid["bust_risk_reduction_pct"] == pytest.approx(0.0, abs=1e-6)
            assert high["bust_risk_reduction_pct"] > 0.0
            assert low["bust_risk_reduction_pct"] < 0.0
            assert low["verdict"] == "Elevated Stopwatch Trap Risk"


def test_real_historical_comps_retrieval_no_mock_data(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    # Querying Day-3 high optical profile must retrieve real cohort prospects from 510 dataset
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
    for bad_pick in (0, -1, 258, 999, True, False, None, "abc", float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rookie_contract_4yr_millions(bad_pick)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            pick_to_round(bad_pick)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fitted_ev_engine.evaluate("WR_TE", bad_pick, 0.5)  # type: ignore[arg-type]
    for bad_oz in (None, True, False, "nan_str", float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            fitted_ev_engine.evaluate("WR_TE", 50, bad_oz)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fitted_ev_engine.evaluate("QUARTERBACK_INVALID", 10, 0.5)
    with pytest.raises(ValueError):
        realized_production_4yr_millions(0.5, 500.0, 10.0, "INVALID_GRP")


def test_python_and_serialized_js_model_params_exact_parity(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    """Verify serialized model_params used by FilmRoom JS match Python DraftSurplusEVEngine.evaluate() 1:1."""
    bundle = fitted_ev_engine.export_ui_and_chart_bundle()
    cba_x = [math.log(p) for p in bundle["cba_pick_anchors"]]
    cba_y = bundle["cba_cost_4yr_m_anchors"]

    def js_eval(grp: str, pick: int, oz: float) -> tuple[float, float, float, str]:
        pm = bundle["model_params"][grp]
        lp = math.log(pick)
        cost = round(float(np.interp(lp, cba_x, cba_y)), 3)
        prod = round(max(2.4, pm["intercept"] + pm["b_log_pick"] * lp + pm["b_optical_z"] * oz), 3)
        surplus = round(prod - cost, 3)
        p_sw = 1.0 / (1.0 + math.exp(-(pm["bust_sw_b0"] + pm["bust_sw_b1"] * lp)))
        p_opt = 1.0 / (1.0 + math.exp(-(pm["bust_opt_b0"] + pm["bust_opt_b1"] * lp + pm["bust_opt_b2"] * oz)))
        bust_red = round(100.0 * (p_sw - p_opt) / max(p_sw, 1e-4), 1)
        if oz <= -0.75 or surplus < -1.5:
            verdict = "Elevated Stopwatch Trap Risk"
        elif surplus >= 12.0 and oz >= 0.75:
            verdict = "High-Conviction Optical Value Target"
        elif surplus >= 5.0 and oz >= 0.0:
            verdict = "Positive Surplus Value Pick"
        else:
            verdict = "Slot-Neutral Market Pick"
        return prod, surplus, bust_red, verdict

    for grp in POSITION_GROUPS:
        for pk, oz in ((1, 1.5), (24, -1.4), (54, 0.5), (177, 0.6), (257, 0.0)):
            py_res = fitted_ev_engine.evaluate(grp, pk, oz)
            j_prod, j_sur, j_br, j_verdict = js_eval(grp, pk, oz)
            assert py_res["expected_production_4yr_m"] == pytest.approx(j_prod, abs=1e-2)
            assert py_res["expected_surplus_ev_m"] == pytest.approx(j_sur, abs=1e-2)
            assert py_res["bust_risk_reduction_pct"] == pytest.approx(j_br, abs=0.11)
            assert py_res["verdict"] == j_verdict


def test_partial_cohort_missing_group_fallback(fitted_ev_engine: DraftSurplusEVEngine) -> None:
    """Fitting on a subset missing OL_LB must gracefully shrink to global prior instead of crashing."""
    partial = fitted_ev_engine.prospects_df[fitted_ev_engine.prospects_df["position_group"] != "OL_LB"].copy()
    eng = DraftSurplusEVEngine.fit(partial)
    res = eng.evaluate("OL_LB", 80, 1.0)
    assert math.isfinite(res["expected_surplus_ev_m"])
    assert res["historical_comps"] == []


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

    fig6_svg = FIGURES_DIR / "fig6_temporal_attention_saliency_and_draft_ev.svg"
    assert fig6_svg.exists()
    fig6_txt = fig6_svg.read_text(encoding="utf-8")
    assert ET.fromstring(fig6_txt).tag.endswith("svg")
    assert "#FFFFFF" in fig6_txt
    fig6_hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", fig6_txt)}
    assert not (fig6_hexes & LEGACY_BEIGE_HEXES)
    for hx in fig6_hexes:
        assert not _is_forbidden_hue(hx), f"Forbidden hue {hx} in fig6 SVG"

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
    assert "#FFFFFF" in html
    html_hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", html)}
    assert not (html_hexes & LEGACY_BEIGE_HEXES)
    for hx in html_hexes:
        assert not _is_forbidden_hue(hx), f"Forbidden hue {hx} in FilmRoom HTML"
    for tok in ("TODO", "FIXME", "PLACEHOLDER"):
        assert tok not in html
