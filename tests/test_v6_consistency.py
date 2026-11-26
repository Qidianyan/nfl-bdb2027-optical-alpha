"""V6 Championship 4-decimal cross-document consistency, Notebook AST, boundary guards, and media deliverable tests."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pandas as pd

from nfl_bdb2027.config import CODE_ROOT, FIGURES_DIR, MATERIALS_DIR
from nfl_bdb2027.report import write_kaggle_writeup_md


def test_v6_four_decimal_consistency_across_json_writeup_readme_and_html() -> None:
    """Every authoritative 4-decimal V4-V6 metric in real_data_metrics_v4_gpu.json must match Writeup, README & FilmRoom HTML."""
    v4 = json.loads((MATERIALS_DIR / "real_data_metrics_v4_gpu.json").read_text(encoding="utf-8"))
    ov = v4["overall_v4_oof"]
    dl = v4["positional_encoders"]["DL_EDGE"]

    required_tokens = [
        f"{ov['cv_r2']:.4f}",                     # 0.2562
        f"{ov['cv_r2_ci95'][0]:.4f}",             # 0.1954
        f"{ov['cv_r2_ci95'][1]:.4f}",             # 0.3150
        f"{ov['temporal_loyo_r2']:.4f}",          # 0.2286
        f"{dl['cv_r2']:.4f}",                     # 0.6892
        f"{dl['temporal_loyo_r2']:.4f}",          # 0.7045
        f"{ov['partial_r2_vs_stopwatch']:.4f}",   # 0.1188
        f"{ov['partial_r2_vs_draft_stopwatch']:.4f}",  # 0.1121
        f"{ov['q_fdr_vs_draft_stopwatch']:.4f}",  # 0.0012
    ]

    writeup_txt = (MATERIALS_DIR / "NFL_BDB2027_Kaggle_Writeup.md").read_text(encoding="utf-8")
    readme_txt = (CODE_ROOT / "README.md").read_text(encoding="utf-8")
    filmroom_txt = (MATERIALS_DIR / "NFL_BDB2027_Interactive_FilmRoom.html").read_text(encoding="utf-8")

    for tok in required_tokens:
        assert tok in writeup_txt, f"Missing 4-decimal V6 metric {tok} in Kaggle Writeup"
        assert tok in readme_txt, f"Missing 4-decimal V6 metric {tok} in README.md"

    # Reject lazy approximations or double-sign bugs in Writeup & README
    assert "+0.73+" not in writeup_txt, "Found lazy approximation '+0.73+' in Writeup"
    assert "+-" not in writeup_txt and "+-" not in readme_txt, "Found malformed '+-' sign formatting"

    # FilmRoom HTML displays core V4-V6 benchmark table metrics
    for tok in (f"{ov['cv_r2']:.4f}", f"{ov['temporal_loyo_r2']:.4f}", f"{dl['cv_r2']:.4f}", f"{dl['temporal_loyo_r2']:.4f}", f"{ov['partial_r2_vs_draft_stopwatch']:.4f}"):
        assert tok in filmroom_txt, f"Missing 4-decimal V6 metric {tok} in Interactive FilmRoom HTML"

    # All 6 2x Retina PNG figures must be embedded in the V6 Writeup and exist on disk
    six_figs = [
        "fig1_kinematic_transfer_matrix.png",
        "fig2_optical_velocity_decel_profiles.png",
        "fig3_draft_capital_residual_alpha.png",
        "fig4_cv_model_ablation.png",
        "fig5_positional_biomechanical_archetypes.png",
        "fig6_temporal_attention_saliency_and_draft_ev.png",
    ]
    for fn in six_figs:
        assert f"figures/{fn}" in writeup_txt, f"Missing figure embed figures/{fn} in Kaggle Writeup"
        assert (FIGURES_DIR / fn).stat().st_size > 20_000


def test_v6_writeup_generator_boundary_and_edge_cases(tmp_path: Path) -> None:
    """write_kaggle_writeup_md must handle empty/missing V4 metrics, empty EV bundles, and negative CIs cleanly."""
    v2 = json.loads((MATERIALS_DIR / "real_data_metrics_v2.json").read_text(encoding="utf-8"))
    v4 = json.loads((MATERIALS_DIR / "real_data_metrics_v4_gpu.json").read_text(encoding="utf-8"))
    inv = v2["dataset_inventory"]
    res = v2["modeling_results"]

    # 1. V2-only fallback (explicit empty v4_metrics={}) must not leak V4 sections or break heading numbers
    p_v2_only = write_kaggle_writeup_md(
        inv, res, pd.DataFrame(), out_path=tmp_path / "v2_only.md", v4_metrics={}, v4_saliency={}, ev_bundle={}
    )
    txt_v2 = p_v2_only.read_text(encoding="utf-8")
    assert "## 5. Position-Specific Biomechanical Archetypes" in txt_v2
    assert "## 6. Reproducibility" in txt_v2
    assert "## 7." not in txt_v2
    assert "### Table 1B" not in txt_v2

    # 2. Negative CI bound + empty ev_bundle curves must never crash or produce '+-' or empty '### B.'
    v4_neg = copy.deepcopy(v4)
    v4_neg["overall_v4_oof"]["partial_r2_vs_draft_stopwatch_ci95"] = [-0.0123, 0.0456]
    p_edge = write_kaggle_writeup_md(
        inv,
        res,
        pd.DataFrame(),
        out_path=tmp_path / "edge.md",
        v4_metrics=v4_neg,
        v4_saliency={},
        ev_bundle={"prospects": [], "curves_by_group": {}},
    )
    txt_edge = p_edge.read_text(encoding="utf-8")
    assert "[-0.0123, +0.0456]" in txt_edge
    assert "+-" not in txt_edge
    assert "### A. Real 0.1s" not in txt_edge
    assert "### B. Realized 4-Year" not in txt_edge


def test_v6_kaggle_notebook_valid_ast_and_v6_scope() -> None:
    """Every code cell in the reproducible Kaggle Notebook must parse via AST, guard optional NPZ, and cover V4-V6 + Draft EV."""
    nb_path = CODE_ROOT / "notebooks" / "nfl_bdb2027_kaggle_notebook.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 7

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 6
    combined_code = []
    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        assert src.strip(), f"Empty code cell {idx}"
        ast.parse(src)
        combined_code.append(src)

    full_src = "\n".join(combined_code)
    assert "real_data_metrics_v4_gpu.json" in full_src
    assert "DraftSurplusEVEngine" in full_src
    assert "write_v5_interactive_filmroom" in full_src
    assert "write_kaggle_writeup_md" in full_src
    # Guard against Kaggle FileNotFoundError / read-only dir / repo-root src lookup
    assert "v4_npz_path.exists()" in full_src
    assert "os.access(MATERIALS_DIR, os.W_OK)" in full_src
    assert 'NB_ROOT / "src"' in full_src


def _parse_srt_ms(ts: str) -> int:
    hh, mm, ss_ms = ts.split(":")
    ss, ms = ss_ms.split(",")
    return ((int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000) + int(ms)


def test_v6_filmroom_reel_and_srt_synced_metrics() -> None:
    """Verify V6 1080p MP4 reel and synced English SRT contain V6 upgraded metrics and strictly monotonic non-overlapping cues."""
    mp4 = MATERIALS_DIR / "NFL_BDB2027_FilmRoom_Reel_1080p.mp4"
    srt = MATERIALS_DIR / "NFL_BDB2027_FilmRoom_Reel_1080p.srt"
    assert mp4.exists() and mp4.stat().st_size > 5_000_000
    assert srt.exists()
    srt_txt = srt.read_text(encoding="utf-8")
    assert "-->" in srt_txt and len(srt_txt) > 300
    for tok in ("0.1560", "0.2398", "0.2562", "0.2286", "0.5432", "0.7045", "0.1121", "0.0012"):
        assert tok in srt_txt, f"Missing 4-decimal V6 metric {tok} in SRT"

    blocks = [b.strip() for b in re.split(r"\n\s*\n", srt_txt.strip()) if b.strip()]
    assert len(blocks) >= 10
    prev_end = -1
    for expected_idx, blk in enumerate(blocks, start=1):
        lines = blk.splitlines()
        assert int(lines[0]) == expected_idx
        start_s, end_s = [part.strip() for part in lines[1].split("-->")]
        st_ms, en_ms = _parse_srt_ms(start_s), _parse_srt_ms(end_s)
        assert st_ms < en_ms, f"Cue {expected_idx} has non-positive duration"
        assert st_ms >= prev_end, f"Cue {expected_idx} overlaps previous cue ({st_ms} < {prev_end})"
        assert any(line.strip() for line in lines[2:]), f"Cue {expected_idx} has empty text"
        prev_end = en_ms
