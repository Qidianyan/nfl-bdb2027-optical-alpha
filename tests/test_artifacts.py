"""Verification tests for V2 Champion artifacts: real_data_metrics_v2.json,
Colab GPU 1D-CNN + Attention OOF fusion, 5 pure-white Economist/NYT SVGs,
forbidden-hue & zero-beige compliance, Canvas 2D 10 Hz Film-Room Replay,
Kaggle Writeup, and Notebook."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from nfl_bdb2027.config import ARTIFACTS_DIR, CODE_ROOT, FIGURES_DIR, MATERIALS_DIR

EXPECTED_SVGS = [
    "fig1_kinematic_transfer_matrix.svg",
    "fig2_optical_velocity_decel_profiles.svg",
    "fig3_draft_capital_residual_alpha.svg",
    "fig4_model_ablation_lift.svg",
    "fig5_positional_biomechanical_archetypes.svg",
]

LEGACY_BEIGE_HEXES = {"#F7F3EA", "#FDFBF7", "#EAE2D3", "#FAF7F0", "#D6CEC0"}


def _is_forbidden_hue(hex_color: str) -> bool:
    """Reject any blue, cyan, teal, or purple hex color (HSL hue 165..315 deg with sat >= 0.14)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < 1e-6:
        return False
    l = (mx + mn) / 2.0
    s = d / (1.0 - abs(2.0 * l - 1.0))
    if s < 0.14:
        return False
    if mx == r:
        hue = (60.0 * ((g - b) / d)) % 360.0
    elif mx == g:
        hue = 60.0 * ((b - r) / d + 2.0)
    else:
        hue = 60.0 * ((r - g) / d + 4.0)
    return 165.0 <= hue <= 315.0


def test_colab_script_and_v2_metrics_json_complete() -> None:
    script_path = CODE_ROOT / "scripts" / "train_temporal_cnn_colab.py"
    assert script_path.exists(), f"Missing {script_path}"
    assert "MultiScale1DCNNTemporalAttention" in script_path.read_text(encoding="utf-8")

    mpath = MATERIALS_DIR / "real_data_metrics_v2.json"
    assert mpath.exists(), f"Missing {mpath}; run `python -m nfl_bdb2027.cli` first."
    data = json.loads(mpath.read_text(encoding="utf-8"))
    inv = data["dataset_inventory"]
    assert inv["total_prospects"] == 510
    assert inv["combine_tracking_player_rows"] > 400_000
    assert inv["game_tracking_total_rows"] > 20_000_000
    assert inv["combine_tracking_coverage"]["players_with_skill_tracking"] == 510

    cnn_meta = data["modeling_results"]["colab_gpu_temporal_cnn"]
    assert cnn_meta.get("trainable_parameters", 0) > 10_000
    assert "L4" in str(cnn_meta.get("gpu_name", "")) or "A100" in str(cnn_meta.get("gpu_name", ""))
    assert (ARTIFACTS_DIR / "colab_temporal_cnn_oof.npz").exists()

    b1 = data["modeling_results"]["benchmark_1_game_kinematic_translation"]
    assert b1["stratified_5fold"]["v2_deep_fusion"]["r2"] > b1["stratified_5fold"]["stopwatch_anthro"]["r2"]
    assert b1["temporal_loyo"]["v2_deep_fusion"]["r2"] > b1["temporal_loyo"]["stopwatch_anthro"]["r2"]
    assert b1["stratified_5fold"]["v2_deep_fusion"]["r2"] > b1["stratified_5fold"]["stopwatch_plus_optical"]["r2"]
    assert b1["temporal_loyo"]["v2_deep_fusion"]["r2"] > b1["temporal_loyo"]["stopwatch_plus_optical"]["r2"]
    assert b1["stratified_5fold"]["v2_deep_fusion"]["r2"] > b1["stratified_5fold"]["optical_only"]["r2"]
    assert b1["temporal_loyo"]["v2_deep_fusion"]["r2"] > b1["temporal_loyo"]["optical_only"]["r2"]


def test_five_svgs_valid_xml_pure_white_and_zero_forbidden_colors() -> None:
    for fname in EXPECTED_SVGS:
        p = FIGURES_DIR / fname
        assert p.exists(), f"Missing figure: {p}"
        txt = p.read_text(encoding="utf-8")
        assert len(txt) > 2500
        root = ET.fromstring(txt)
        assert root.tag.endswith("svg")
        assert "#FFFFFF" in txt, f"Pure-white ground #FFFFFF missing in {fname}"
        hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", txt)}
        assert not (hexes & LEGACY_BEIGE_HEXES), f"Legacy warm beige {hexes & LEGACY_BEIGE_HEXES} found in {fname}"
        for hx in hexes:
            assert not _is_forbidden_hue(hx), f"Forbidden blue/cyan/teal/purple color {hx} found in {fname}"


def test_writeup_html_filmroom_notebook_and_no_placeholders() -> None:
    md_path = MATERIALS_DIR / "NFL_BDB2027_Kaggle_Writeup.md"
    html_path = MATERIALS_DIR / "NFL_BDB2027_Interactive_Report.html"
    nb_path = CODE_ROOT / "notebooks" / "nfl_bdb2027_kaggle_notebook.ipynb"

    for p in (md_path, html_path, nb_path):
        assert p.exists(), f"Missing deliverable: {p}"
        txt = p.read_text(encoding="utf-8")
        assert len(txt) > 4000
        assert not re.search(r"\b(TODO|FIXME|PLACEHOLDER)\b", txt)

    html_txt = html_path.read_text(encoding="utf-8")
    assert "Puka Nacua" in html_txt
    assert "Ja'Lynn Polk" in html_txt
    assert "Jaylon Jones" in html_txt
    assert "<svg" in html_txt
    assert 'id="fieldCanvas"' in html_txt
    assert 'id="telemetryCanvas"' in html_txt
    assert "#FFFFFF" in html_txt
    assert "#111111" in html_txt
    assert "#184E32" in html_txt
    assert "#CC2929" in html_txt
    hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", html_txt)}
    assert not (hexes & LEGACY_BEIGE_HEXES), f"Legacy warm beige {hexes & LEGACY_BEIGE_HEXES} in HTML report"
    for hx in hexes:
        assert not _is_forbidden_hue(hx), f"Forbidden blue/cyan/teal/purple color {hx} in HTML report"

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
