"""V6 Championship 4-decimal cross-document consistency, Notebook AST, and media deliverable tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from nfl_bdb2027.config import CODE_ROOT, FIGURES_DIR, MATERIALS_DIR


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


def test_v6_kaggle_notebook_valid_ast_and_v6_scope() -> None:
    """Every code cell in the reproducible Kaggle Notebook must parse via AST and cover V4-V6 + Draft EV."""
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


def test_v6_filmroom_reel_and_srt_synced_metrics() -> None:
    """Verify V6 1080p MP4 reel and synced English SRT contain V6 upgraded metrics."""
    mp4 = MATERIALS_DIR / "NFL_BDB2027_FilmRoom_Reel_1080p.mp4"
    srt = MATERIALS_DIR / "NFL_BDB2027_FilmRoom_Reel_1080p.srt"
    assert mp4.exists() and mp4.stat().st_size > 5_000_000
    assert srt.exists()
    srt_txt = srt.read_text(encoding="utf-8")
    assert "-->" in srt_txt and len(srt_txt) > 300
    assert "0.2562" in srt_txt
    assert "0.2286" in srt_txt
    assert "0.7045" in srt_txt
