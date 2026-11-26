"""V5 Economist/NYT Pure-White Interactive Film-Room Studio & GM Draft-Room EV Simulator.

Generates `materials/NFL_BDB2027_Interactive_FilmRoom.html` combining:
  1. Synchronized 10 Hz Combine Ghost Comparison + 0.1s Dual-Head Temporal Attention
     glowing saliency overlays across all 4 position-group matchups (100% real tracking
     from `combine_tracking.csv` + OOF Colab L4 GPU weights from `v4_saliency_profiles.json`).
  2. Interactive GM Draft-Room Surplus EV Calculator calibrated on all 510 real prospects
     and official NFL CBA rookie wage scale anchors (`draft_ev.py`).
  3. Zero-leakage V4 statistical rigor benchmark table & 6-figure 2x Retina editorial gallery.

Strict pure-white Economist/NYT editorial palette:
#FFFFFF ground, #111111 ink, #184E32 NFL turf green, #C43C29 crimson, #C68B2C warm gold.
Zero warm beige/parchment; zero blue, deep-teal, cyan, or purple hues.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .config import MATERIALS_DIR
from .data import DatasetPaths, scan_combine_tracking
from .kinematics import compute_frame_kinematics
from .replay import _extract_best_combine_rep


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=True)


def _safe_json_for_script(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def build_v5_saliency_matchups_with_xy(
    paths: DatasetPaths,
    v4_saliency: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pair every V4 0.1s saliency matchup with true 10 Hz `(x, y)` stance-normalized trajectories."""
    matchups_raw = v4_saliency["matchups"]
    target_ids = sorted(
        {int(m[side]["nfl_id"]) for m in matchups_raw for side in ("gem", "trap")}
    )

    ct_df = (
        compute_frame_kinematics(scan_combine_tracking(paths))
        .filter(pl.col("nfl_id").is_in(target_ids))
        .select(["nfl_id", "canon_drill", "event_id", "attempt", "frame_idx", "x", "y", "s"])
        .collect()
    )

    enriched: list[dict[str, Any]] = []
    short_labels = {
        "WR_TE": "WR/TE",
        "DB": "DB",
        "DL_EDGE": "DL/EDGE",
        "OL_LB": "OL",
    }

    for m in matchups_raw:
        grp = m["position_group"]
        sides_out: dict[str, Any] = {}
        raw_dx_dy: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for side in ("gem", "trap"):
            p = m[side]
            rep = _extract_best_combine_rep(ct_df, int(p["nfl_id"]), str(p["drill_name"]))
            n_sal = int(p["n_frames"])
            if len(rep) >= n_sal:
                rep_use = rep.iloc[:n_sal].reset_index(drop=True)
                x_arr = rep_use["x"].to_numpy(dtype=float)
                y_arr = rep_use["y"].to_numpy(dtype=float)
            else:
                idx_grid = np.linspace(0, len(rep) - 1, n_sal)
                x_arr = np.interp(idx_grid, np.arange(len(rep)), rep["x"].to_numpy(dtype=float))
                y_arr = np.interp(idx_grid, np.arange(len(rep)), rep["y"].to_numpy(dtype=float))
            raw_dx_dy[side] = (x_arr - x_arr[0], y_arr - y_arr[0])

        gem_dx, gem_dy = raw_dx_dy["gem"]
        trap_dx, trap_dy = raw_dx_dy["trap"]
        if (gem_dy[-1] * trap_dy[-1]) < -4.0:
            trap_dy = -trap_dy
        if (gem_dx[-1] * trap_dx[-1]) < -4.0:
            trap_dx = -trap_dx
        raw_dx_dy["gem"] = (gem_dx, gem_dy)
        raw_dx_dy["trap"] = (trap_dx, trap_dy)

        for side in ("gem", "trap"):
            p = m[side]
            dx, dy = raw_dx_dy[side]
            frames_out = []
            for fi, fr in enumerate(p["frames"]):
                frames_out.append(
                    {
                        "t": round(float(fr["t_sec"]), 1),
                        "dx": round(float(dx[fi]), 2),
                        "dy": round(float(dy[fi]), 2),
                        "s": round(float(fr["speed_yd_s"]), 2),
                        "a": round(float(fr["a_tan_yd_s2"]), 2),
                        "w": round(float(fr["omega_rad_s"]), 2),
                        "ac": round(float(fr["a_cent_yd_s2"]), 2),
                        "aw": round(float(fr["attn_weight"]), 5),
                        "h1": round(float(fr["attn_head1_burst"]), 5),
                        "h2": round(float(fr["attn_head2_brake"]), 5),
                    }
                )
            rd = p.get("draft_round")
            pk_num = p.get("draft_overall_pick")
            draft_str = f"Rd {int(rd)}, Pick #{int(pk_num)}" if rd is not None and pk_num is not None else "UDFA"
            f40 = p.get("forty_sec")
            forty_str = "40-yd opt-out" if f40 is None else f"{float(f40):.2f}s 40-yd"

            sides_out[side] = {
                "nfl_id": int(p["nfl_id"]),
                "name": str(p["display_name"]),
                "draft_year": int(p["draft_year"]),
                "pos": str(p["combine_position"]),
                "draft_label": draft_str,
                "forty_label": forty_str,
                "alpha": round(float(p["draft_capital_alpha"]), 2),
                "snaps": round(float(p["snaps_per_season"]), 0),
                "drill_name": str(p["drill_name"]),
                "rep_mil_weight": round(float(p["rep_mil_weight"]), 4),
                "peak_win": p["peak_0_3s_window"],
                "frames": frames_out,
            }

        short_titles = {
            "WR_TE": "Gauntlet drill: 0.3s braking and cut attention.",
            "DB": "90-degree break: hip-flip deceleration attention.",
            "DL_EDGE": "Pass-rush drill: first-step burst attention.",
            "OL_LB": "Pass-pro mirror: lateral anchor balance attention.",
        }
        clean_title = short_titles.get(grp, str(m["title"]).replace("–", " to ").replace("—", ", "))
        enriched.append(
            {
                "id": str(m["matchup_id"]),
                "group": grp,
                "tab_label": f"{short_labels.get(grp, grp)}: {sides_out['gem']['name'].split()[-1]} vs {sides_out['trap']['name'].split()[-1]}",
                "title": clean_title,
                "drill_pretty": str(sides_out["gem"]["drill_name"]).replace("_", " ").title(),
                "cohort_window": v4_saliency["cohort_position_saliency"][grp],
                "gem": sides_out["gem"],
                "trap": sides_out["trap"],
            }
        )

    return enriched


def write_v5_interactive_filmroom(
    paths: DatasetPaths,
    v4_metrics: dict[str, Any],
    v4_saliency: dict[str, Any],
    ev_bundle: dict[str, Any],
    out_path: Path = MATERIALS_DIR / "NFL_BDB2027_Interactive_FilmRoom.html",
) -> Path:
    """Render the complete V5 Economist/NYT Interactive Film-Room Studio HTML."""
    matchups = build_v5_saliency_matchups_with_xy(paths, v4_saliency)
    ov = v4_metrics["overall_v4_oof"]
    sw = v4_metrics["stopwatch_baseline"]
    v2 = v4_metrics["v2_baseline"]
    pos_enc = v4_metrics["positional_encoders"]

    payload = {
        "matchups": matchups,
        "ev_bundle": ev_bundle,
        "v4_summary": {
            "overall_r2": ov["cv_r2"],
            "overall_loyo_r2": ov["temporal_loyo_r2"],
            "dl_edge_r2": pos_enc["DL_EDGE"]["cv_r2"],
            "dl_edge_loyo_r2": pos_enc["DL_EDGE"]["temporal_loyo_r2"],
        },
    }
    json_blob = _safe_json_for_script(payload)

    tab_buttons_html = []
    for idx, m in enumerate(matchups):
        sel = "true" if idx == 0 else "false"
        tidx = "0" if idx == 0 else "-1"
        tab_buttons_html.append(
            f'<button type="button" class="matchup-tab" role="tab" id="tab-{_esc(m["id"])}" '
            f'data-matchup-idx="{idx}" aria-selected="{sel}" tabindex="{tidx}">'
            f'{_esc(m["tab_label"])}</button>'
        )
    tabs_markup = "\n          ".join(tab_buttons_html)

    rigor_rows_data = [
        (
            "Overall V4 fusion",
            f"n={ov['n']}",
            f"+{sw['cv_r2']:.4f}",
            f"+{v2['cv_r2']:.4f}",
            f"+{ov['cv_r2']:.4f}",
            f"+{ov['temporal_loyo_r2']:.4f}",
            f"+{ov['partial_r2_vs_stopwatch']:.4f}",
            f"{ov['q_fdr_vs_stopwatch']:.4f}",
            True,
        ),
        (
            "DL/EDGE encoder",
            f"n={pos_enc['DL_EDGE']['n_players']}",
            f"+{pos_enc['DL_EDGE']['stopwatch_r2']:.4f}",
            f"+{pos_enc['DL_EDGE']['v2_unified_r2']:.4f}",
            f"+{pos_enc['DL_EDGE']['cv_r2']:.4f}",
            f"+{pos_enc['DL_EDGE']['temporal_loyo_r2']:.4f}",
            f"+{pos_enc['DL_EDGE']['partial_r2_vs_stopwatch']:.4f}",
            f"{pos_enc['DL_EDGE']['q_fdr_vs_stopwatch']:.4f}",
            True,
        ),
        (
            "WR/TE encoder",
            f"n={pos_enc['WR_TE']['n_players']}",
            f"+{pos_enc['WR_TE']['stopwatch_r2']:.4f}",
            f"+{pos_enc['WR_TE']['v2_unified_r2']:.4f}",
            f"+{pos_enc['WR_TE']['cv_r2']:.4f}",
            f"+{pos_enc['WR_TE']['temporal_loyo_r2']:.4f}",
            f"+{pos_enc['WR_TE']['partial_r2_vs_stopwatch']:.4f}",
            f"{pos_enc['WR_TE']['q_fdr_vs_stopwatch']:.4f}",
            False,
        ),
        (
            "DB encoder",
            f"n={pos_enc['DB']['n_players']}",
            f"{pos_enc['DB']['stopwatch_r2']:+.4f}",
            f"+{pos_enc['DB']['v2_unified_r2']:.4f}",
            f"+{pos_enc['DB']['cv_r2']:.4f}",
            f"{pos_enc['DB']['temporal_loyo_r2']:+.4f}",
            f"+{pos_enc['DB']['partial_r2_vs_stopwatch']:.4f}",
            f"{pos_enc['DB']['q_fdr_vs_stopwatch']:.4f}",
            True,
        ),
        (
            "OL encoder",
            f"n={pos_enc['OL_LB']['n_players']}",
            f"{pos_enc['OL_LB']['stopwatch_r2']:+.4f}",
            f"{pos_enc['OL_LB']['v2_unified_r2']:+.4f}",
            f"{pos_enc['OL_LB']['cv_r2']:+.4f}",
            f"{pos_enc['OL_LB']['temporal_loyo_r2']:+.4f}",
            f"+{pos_enc['OL_LB']['partial_r2_vs_stopwatch']:.4f}",
            f"{pos_enc['OL_LB']['q_fdr_vs_stopwatch']:.4f}",
            False,
        ),
    ]

    rigor_rows_html = []
    for name, n_str, sw_s, v2_s, v4_s, loyo_s, pr2_s, q_s, sig in rigor_rows_data:
        row_cls = ' class="row-highlight"' if sig else ""
        rigor_rows_html.append(
            f"<tr{row_cls}>"
            f'<th scope="row" data-label="Encoder">{_esc(name)}</th>'
            f'<td class="num" data-label="Cohort">{_esc(n_str)}</td>'
            f'<td class="num" data-label="Stopwatch">{_esc(sw_s)}</td>'
            f'<td class="num" data-label="V2">{_esc(v2_s)}</td>'
            f'<td class="num num-strong" data-label="V4 OOF">{_esc(v4_s)}</td>'
            f'<td class="num num-strong" data-label="LOYO">{_esc(loyo_s)}</td>'
            f'<td class="num num-turf" data-label="Partial R²">{_esc(pr2_s)}</td>'
            f'<td class="num" data-label="q_FDR">{_esc(q_s)}</td>'
            "</tr>"
        )
    rigor_tbody = "\n            ".join(rigor_rows_html)

    fig_gallery = [
        (
            "figures/fig6_temporal_attention_saliency_and_draft_ev.png",
            1180,
            732,
            "Fig 6: Attention and Surplus EV",
        ),
        (
            "figures/fig4_cv_model_ablation.png",
            1180,
            692,
            "Fig 4: V4 Ablation Lift",
        ),
        (
            "figures/fig1_kinematic_transfer_matrix.png",
            1180,
            728,
            "Fig 1: Transfer Matrix",
        ),
        (
            "figures/fig2_optical_velocity_decel_profiles.png",
            1180,
            640,
            "Fig 2: Velocity and Braking",
        ),
        (
            "figures/fig3_draft_capital_residual_alpha.png",
            1180,
            660,
            "Fig 3: Draft Residual Alpha",
        ),
        (
            "figures/fig5_positional_biomechanical_archetypes.png",
            1180,
            704,
            "Fig 5: Positional Archetypes",
        ),
    ]
    fig_cards_html = []
    for src, fw, fh, title in fig_gallery:
        fig_cards_html.append(
            f'<figure class="fig-card">'
            f'<a href="{_esc(src)}" target="_blank" rel="noopener noreferrer" class="fig-link">'
            f'<img src="{_esc(src)}" alt="{_esc(title)}" width="{fw}" height="{fh}" loading="lazy">'
            f"</a>"
            f'<figcaption class="fig-cap">'
            f'<span class="fig-title">{_esc(title)}</span>'
            f"</figcaption>"
            f"</figure>"
        )
    gallery_markup = "\n        ".join(fig_cards_html)

    first = matchups[0]
    first_gem = first["gem"]
    first_trap = first["trap"]
    first_fmax = max(len(first_gem["frames"]), len(first_trap["frames"])) - 1

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Interactive 10 Hz NFL Combine optical film room and GM draft-room 4-year rookie surplus value calculator calibrated on 510 Big Data Bowl 2027 prospects.">
  <title>NFL Big Data Bowl 2027: Optical Film-Room and GM Surplus Studio</title>
  <style>
    :root {{
      --bg-paper: #FFFFFF;
      --bg-subtle: #F7F7F7;
      --bg-tint: #F2F2F2;
      --border-hairline: #E2E2E2;
      --border-track: #D4D4D4;
      --border-strong: #111111;
      --ink: #111111;
      --ink-secondary: #4A4A4A;
      --ink-muted: #666666;
      --brand-green: #184E32;
      --brand-green-mid: #2E6F4E;
      --brand-green-tint: #EAF2EE;
      --accent-crimson: #C43C29;
      --accent-crimson-tint: #FCEEEE;
      --accent-gold: #C68B2C;
      --accent-gold-ink: #875E12;
      --accent-gold-tint: #FBF4E4;

      --font-serif: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      --font-sans: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;

      --text-sm: 0.8125rem;
      --text-body: 0.9375rem;
      --text-heading: 1.75rem;
      --radius-sm: 3px;
      --section-gap: 4.5rem;
    }}

    *, *::before, *::after {{
      box-sizing: border-box;
    }}

    html {{
      color-scheme: light;
      overflow-x: clip;
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      background-color: var(--bg-paper);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: var(--text-body);
      line-height: 1.55;
      accent-color: var(--brand-green);
      overflow-x: clip;
      -webkit-font-smoothing: antialiased;
    }}

    p, figcaption {{
      text-wrap: pretty;
    }}

    h1, h2, h3 {{
      text-wrap: balance;
    }}

    .skip-link {{
      position: absolute;
      inset-inline-start: 1rem;
      inset-block-start: -4rem;
      background-color: var(--ink);
      color: var(--bg-paper);
      padding: 0.6rem 1rem;
      font-size: var(--text-sm);
      font-weight: 600;
      text-decoration: none;
      z-index: 100;
    }}

    .skip-link:focus {{
      inset-block-start: 0.75rem;
    }}

    :focus-visible {{
      outline: 2px solid var(--brand-green);
      outline-offset: 2px;
    }}

    .num {{
      font-variant-numeric: tabular-nums;
    }}

    .page-shell {{
      max-width: 1220px;
      margin-inline: auto;
      padding-inline: 1.5rem;
    }}

    /* Crisp Economist Masthead */
    .masthead {{
      border-block-end: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      position: sticky;
      inset-block-start: 0;
      z-index: 40;
    }}

    .masthead-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.25rem;
      padding-block: 0.75rem;
      border-block-start: 4px solid var(--accent-crimson);
    }}

    .brand-name {{
      font-family: var(--font-serif);
      font-size: 1.125rem;
      font-weight: 700;
      color: var(--ink);
      letter-spacing: -0.01em;
      white-space: nowrap;
    }}

    .top-nav {{
      display: flex;
      align-items: center;
      gap: 0.95rem;
      flex-wrap: nowrap;
      overflow-x: auto;
    }}

    .top-nav a {{
      color: var(--ink-secondary);
      text-decoration: none;
      font-size: var(--text-sm);
      font-weight: 600;
      white-space: nowrap;
      flex: none;
      padding-block: 0.25rem;
      border-block-end: 2px solid transparent;
      transition: color 140ms ease, border-color 140ms ease;
    }}

    .top-nav a:hover {{
      color: var(--ink);
      border-block-end-color: var(--accent-crimson);
    }}

    .hero-link {{
      color: var(--ink);
      font-size: var(--text-sm);
      font-weight: 600;
      text-decoration: underline;
      text-underline-offset: 4px;
      white-space: nowrap;
      padding-block: 0.5rem;
      transition: color 140ms ease;
    }}

    .hero-link:hover {{
      color: var(--accent-crimson);
    }}

    /* Compact Editorial Lead directly into Live Film-Room */
    .editorial-header {{
      padding-block: 1.75rem 1.35rem;
      border-block-end: 1px solid var(--border-hairline);
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      justify-content: space-between;
      gap: 1rem 1.5rem;
    }}

    .editorial-title-wrap {{
      max-width: 720px;
    }}

    h1 {{
      margin: 0 0 0.4rem 0;
      font-family: var(--font-serif);
      font-size: clamp(1.75rem, 2.5vw, 2.25rem);
      line-height: 1.15;
      letter-spacing: -0.015em;
      color: var(--ink);
    }}

    .editorial-deck {{
      margin: 0;
      font-size: var(--text-body);
      color: var(--ink-secondary);
    }}

    .header-actions {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }}

    /* Sections */
    .studio-section {{
      scroll-margin-top: 64px;
      padding-block: 2rem var(--section-gap);
    }}

    .studio-section + .studio-section {{
      padding-block-start: var(--section-gap);
      border-block-start: 2px solid var(--border-strong);
    }}

    .section-head {{
      margin-block-end: 1.25rem;
    }}

    h2 {{
      margin: 0;
      font-family: var(--font-serif);
      font-size: var(--text-heading);
      line-height: 1.2;
      letter-spacing: -0.01em;
      color: var(--ink);
    }}

    .section-lead {{
      margin: 0.35rem 0 0 0;
      font-size: var(--text-body);
      color: var(--ink-secondary);
      max-width: 68ch;
    }}

    /* Accessible single-row horizontal tablist */
    .tablist-row {{
      display: flex;
      flex-wrap: nowrap;
      overflow-x: auto;
      gap: 0.25rem;
      border-block-end: 1px solid var(--border-hairline);
      margin-block-end: 1.15rem;
      scroll-snap-type: inline proximity;
    }}

    .matchup-tab {{
      position: relative;
      flex: none;
      scroll-snap-align: start;
      appearance: none;
      background: transparent;
      border: 0;
      min-block-size: 44px;
      padding: 0.65rem 1rem;
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--ink-secondary);
      cursor: pointer;
      white-space: nowrap;
      transition: color 140ms ease, background-color 140ms ease;
    }}

    .matchup-tab:hover {{
      color: var(--ink);
      background-color: var(--bg-subtle);
    }}

    .matchup-tab:active {{
      background-color: var(--bg-tint);
    }}

    .matchup-tab[aria-selected="true"] {{
      color: var(--ink);
    }}

    .matchup-tab[aria-selected="true"]::after {{
      content: "";
      position: absolute;
      inset: auto 0 0 0;
      border-block-end: 3px solid var(--accent-crimson);
    }}

    /* Matchup Comparison Header Strip (hairline all around + crisp square swatch, zero side stripe) */
    .matchup-brief {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 1rem;
      margin-block-end: 1.15rem;
    }}

    .player-strip {{
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      padding: 0.85rem 1rem;
    }}

    .player-strip-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
    }}

    .player-name-wrap {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .swatch {{
      width: 10px;
      height: 10px;
      display: inline-block;
      flex: none;
      background-color: var(--brand-green);
    }}

    .swatch-trap {{
      background-color: var(--accent-crimson);
    }}

    .player-name {{
      font-family: var(--font-serif);
      font-size: 1.0625rem;
      font-weight: 700;
      color: var(--ink);
    }}

    .player-role {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--brand-green);
    }}

    .player-role-trap {{
      color: var(--accent-crimson);
    }}

    .player-meta {{
      margin-block-start: 0.25rem;
      font-size: var(--text-sm);
      color: var(--ink-secondary);
    }}

    /* Film-Room Canvas Workspace */
    .filmroom-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr);
      gap: 1.25rem;
      align-items: start;
    }}

    .canvas-panel {{
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      padding: 0.9rem;
    }}

    .canvas-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.75rem;
      margin-block-end: 0.6rem;
      flex-wrap: wrap;
    }}

    .canvas-title {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--ink);
      margin: 0;
    }}

    .canvas-hint {{
      font-size: var(--text-sm);
      color: var(--ink-muted);
    }}

    .studio-canvas {{
      width: 100%;
      height: auto;
      display: block;
      background-color: var(--bg-paper);
      border: 1px solid var(--border-hairline);
      cursor: crosshair;
    }}

    /* Transport & Attention Head Controls: crisp single-row 3-col grid at desktop, stacked on mobile */
    .control-bar {{
      margin-block-start: 1rem;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-subtle);
      display: grid;
      grid-template-columns: auto auto minmax(200px, 1fr);
      align-items: center;
      gap: 0.75rem 1.25rem;
    }}

    .transport-group {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
    }}

    .btn {{
      appearance: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-block-size: 44px;
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 600;
      line-height: 1.2;
      padding: 0.65rem 1rem;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-strong);
      background-color: var(--bg-paper);
      color: var(--ink);
      text-decoration: none;
      white-space: nowrap;
      cursor: pointer;
      transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
    }}

    .btn:hover {{
      background-color: var(--bg-tint);
    }}

    .btn:active {{
      background-color: var(--border-hairline);
    }}

    .btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}

    .btn-primary {{
      background-color: var(--brand-green);
      border-color: var(--brand-green);
      color: var(--bg-paper);
    }}

    .btn-primary:hover {{
      background-color: var(--brand-green-mid);
      border-color: var(--brand-green-mid);
    }}

    .btn-primary:active {{
      background-color: var(--ink);
      border-color: var(--ink);
    }}

    .head-fieldset {{
      margin: 0;
      padding: 0;
      border: 0;
      display: flex;
      align-items: center;
      gap: 0.85rem;
      flex-wrap: wrap;
    }}

    .head-legend {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--ink);
      margin-inline-end: 0.3rem;
      float: left;
    }}

    .radio-label {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      min-block-size: 36px;
      font-size: var(--text-sm);
      color: var(--ink);
      cursor: pointer;
    }}

    .scrubber-wrap {{
      flex: 1 1 220px;
      display: flex;
      align-items: center;
      gap: 0.75rem;
      min-width: 200px;
    }}

    .scrubber-label {{
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--ink);
      white-space: nowrap;
    }}

    .scrubber-output {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--brand-green);
      min-width: 6.5rem;
      text-align: right;
    }}

    /* Brand-consistent accessible range sliders (strictly token-backed) */
    input[type="range"] {{
      --slider-track-size: 0.375rem;
      --slider-thumb-size: 1.15rem;
      --slider-track-radius: 999px;
      --slider-thumb-radius: 50%;
      --slider-track-color: var(--border-track);
      --slider-fill-color: var(--brand-green);
      --slider-thumb-color: var(--brand-green);
      --slider-direction: to right;
      --attr-min: 0;
      --attr-max: 100;
      --control-value: 0;
      --slider-progress: calc(
        progress(var(--control-value), var(--attr-min), var(--attr-max)) * 100%
      );

      appearance: none;
      -webkit-appearance: none;
      inline-size: 100%;
      block-size: 2.75rem;
      margin: 0;
      padding: 0;
      border: 0;
      background: transparent;
      cursor: pointer;
    }}

    @supports not (opacity: progress(1, 0, 100)) {{
      input[type="range"] {{
        --slider-progress: calc(
          (var(--control-value) - var(--attr-min)) /
          (var(--attr-max) - var(--attr-min)) * 100%
        );
      }}
    }}

    input[type="range"]::-webkit-slider-runnable-track {{
      block-size: var(--slider-track-size);
      border: 0;
      border-radius: var(--slider-track-radius);
      background: linear-gradient(
        var(--slider-direction),
        var(--slider-fill-color) var(--slider-progress),
        var(--slider-track-color) var(--slider-progress)
      );
    }}

    input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      inline-size: var(--slider-thumb-size);
      block-size: var(--slider-thumb-size);
      margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
      border: 2px solid var(--bg-paper);
      border-radius: var(--slider-thumb-radius);
      background-color: var(--slider-thumb-color);
    }}

    input[type="range"]::-moz-range-track {{
      block-size: var(--slider-track-size);
      border: 0;
      border-radius: var(--slider-track-radius);
      background-color: var(--slider-track-color);
    }}

    input[type="range"]::-moz-range-progress {{
      block-size: var(--slider-track-size);
      border-radius: var(--slider-track-radius);
      background-color: var(--slider-fill-color);
    }}

    input[type="range"]::-moz-range-thumb {{
      inline-size: var(--slider-thumb-size);
      block-size: var(--slider-thumb-size);
      border: 2px solid var(--bg-paper);
      border-radius: var(--slider-thumb-radius);
      background-color: var(--slider-thumb-color);
    }}

    input[type="range"]:focus-visible {{
      outline: 2px solid var(--brand-green);
      outline-offset: 3px;
    }}

    /* Live 0.1s Telemetry & Benchmark Tables (restack cleanly on mobile so zero columns overflow) */
    .telemetry-wrap,
    .table-scroll {{
      margin-block-start: 1rem;
      border: 1px solid var(--border-hairline);
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: var(--text-sm);
    }}

    .data-table caption {{
      text-align: left;
      padding: 0.65rem 0.9rem;
      font-weight: 700;
      color: var(--ink);
      background-color: var(--bg-subtle);
      border-block-end: 1px solid var(--border-hairline);
    }}

    .data-table th,
    .data-table td {{
      padding: 0.62rem 0.78rem;
      border-block-end: 1px solid var(--border-hairline);
      text-align: left;
    }}

    .data-table thead th {{
      font-weight: 700;
      color: var(--ink-secondary);
      background-color: var(--bg-paper);
      border-block-end: 2px solid var(--border-strong);
    }}

    .data-table th.num,
    .data-table td.num {{
      text-align: right;
    }}

    .num-strong {{
      font-weight: 700;
      color: var(--ink);
    }}

    .num-turf {{
      font-weight: 700;
      color: var(--brand-green);
    }}

    .num-crimson {{
      font-weight: 700;
      color: var(--accent-crimson);
    }}

    .num-gold {{
      font-weight: 700;
      color: var(--accent-gold-ink);
    }}

    .row-highlight {{
      background-color: var(--brand-green-tint);
    }}

    /* Module 2: GM Draft Room Surplus EV Calculator (zero nested card-in-card!) */
    .ev-workspace {{
      display: grid;
      grid-template-columns: minmax(280px, 0.88fr) minmax(0, 1.5fr);
      gap: 2rem;
      align-items: start;
    }}

    .ev-controls-col {{
      padding-inline-end: 1.5rem;
      border-inline-end: 1px solid var(--border-hairline);
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}

    .ev-fieldset {{
      margin: 0;
      padding: 0;
      border: 0;
    }}

    .ev-fieldset legend {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--ink);
      margin-block-end: 0.55rem;
    }}

    .pos-radio-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.45rem;
    }}

    .pos-radio-item {{
      display: flex;
      align-items: center;
      gap: 0.45rem;
      min-block-size: 44px;
      padding: 0.5rem 0.65rem;
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      font-size: var(--text-sm);
      font-weight: 600;
      color: var(--ink);
      cursor: pointer;
      transition: border-color 140ms ease, background-color 140ms ease;
    }}

    .pos-radio-item:has(input:checked) {{
      border-color: var(--brand-green);
      background-color: var(--brand-green-tint);
      color: var(--brand-green);
    }}

    .slider-field {{
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }}

    .slider-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.5rem;
    }}

    .slider-header label {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--ink);
    }}

    .slider-header output {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--brand-green);
    }}

    .slider-scale {{
      display: flex;
      justify-content: space-between;
      font-size: var(--text-sm);
      color: var(--ink-muted);
    }}

    .preset-stack {{
      border-block-start: 1px solid var(--border-hairline);
      padding-block-start: 0.9rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .preset-heading {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--ink);
      margin: 0 0 0.1rem 0;
    }}

    .btn-preset {{
      width: 100%;
      justify-content: flex-start;
      padding: 0.55rem 0.75rem;
      border-color: var(--border-hairline);
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .btn-preset:hover {{
      border-color: var(--border-strong);
    }}

    /* EV Output Cards & Curve Canvas */
    .ev-output-col {{
      display: flex;
      flex-direction: column;
      gap: 1.15rem;
    }}

    .ev-kpi-row {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.85rem;
    }}

    .ev-kpi {{
      border: 1px solid var(--border-hairline);
      border-block-start: 3px solid var(--brand-green);
      background-color: var(--bg-paper);
      padding: 0.85rem 0.95rem;
    }}

    .ev-kpi-crimson {{
      border-block-start-color: var(--accent-crimson);
    }}

    .ev-kpi-gold {{
      border-block-start-color: var(--accent-gold);
    }}

    .ev-kpi-label {{
      font-size: var(--text-sm);
      color: var(--ink-muted);
      display: block;
    }}

    .ev-kpi-val {{
      font-family: var(--font-serif);
      font-size: 1.55rem;
      font-weight: 700;
      line-height: 1.2;
      color: var(--ink);
      margin-block: 0.2rem;
      display: block;
    }}

    .ev-kpi-sub {{
      font-size: var(--text-sm);
      color: var(--ink-secondary);
      display: block;
    }}

    .verdict-strip {{
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-subtle);
      padding: 0.75rem 0.95rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem 1rem;
      font-size: var(--text-sm);
    }}

    .verdict-strip-risk {{
      background-color: var(--accent-crimson-tint);
      border-color: var(--accent-crimson);
    }}

    .verdict-label-wrap {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .comps-grid {{
      margin-block-start: 0.65rem;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.85rem;
    }}

    .comp-card {{
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      padding: 0.85rem 0.9rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }}

    .comp-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.5rem;
    }}

    .comp-name {{
      font-family: var(--font-serif);
      font-size: 0.96rem;
      font-weight: 700;
      color: var(--ink);
    }}

    .comp-sim {{
      font-size: var(--text-sm);
      font-weight: 700;
      color: var(--brand-green);
    }}

    .comp-meta {{
      font-size: var(--text-sm);
      color: var(--ink-secondary);
    }}

    .comp-stats {{
      margin-block-start: 0.25rem;
      padding-block-start: 0.45rem;
      border-block-start: 1px solid var(--border-hairline);
      display: flex;
      justify-content: space-between;
      font-size: var(--text-sm);
      color: var(--ink);
    }}

    /* Module 4: 6 Figures */
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.25rem;
    }}

    .fig-card {{
      margin: 0;
      border: 1px solid var(--border-hairline);
      background-color: var(--bg-paper);
      display: flex;
      flex-direction: column;
    }}

    .fig-link {{
      display: block;
      background-color: var(--bg-paper);
    }}

    .fig-card img {{
      display: block;
      width: 100%;
      height: auto;
      border-block-end: 1px solid var(--border-hairline);
    }}

    .fig-cap {{
      padding: 0.7rem 0.9rem;
    }}

    .fig-title {{
      font-family: var(--font-serif);
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--ink);
    }}

    .site-footer {{
      border-block-start: 2px solid var(--border-strong);
      padding-block: 1.5rem 2.5rem;
      font-size: var(--text-sm);
      color: var(--ink-muted);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 0.75rem;
    }}

    @media (max-width: 980px) {{
      .control-bar,
      .filmroom-grid,
      .ev-workspace,
      .matchup-brief,
      .gallery-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .ev-controls-col {{
        padding-inline-end: 0;
        border-inline-end: 0;
        padding-block-end: 1.25rem;
        border-block-end: 1px solid var(--border-hairline);
      }}
    }}

    @media (max-width: 680px) {{
      .page-shell {{
        padding-inline: 1rem;
      }}
      .ev-kpi-row,
      .comps-grid {{
        grid-template-columns: minmax(0, 1fr);
      }}
      /* Restack tables into vertical 2-column rows on narrow screens: zero off-screen columns */
      .data-table thead {{
        display: none;
      }}
      .data-table,
      .data-table tbody,
      .data-table tr,
      .data-table th,
      .data-table td {{
        display: block;
        width: 100%;
      }}
      .data-table tr {{
        padding: 0.65rem 0.8rem;
        border-block-end: 2px solid var(--border-hairline);
      }}
      .data-table tbody th,
      .data-table tbody td {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        padding: 0.25rem 0;
        border-block-end: 0;
        text-align: right;
      }}
      .data-table tbody th::before,
      .data-table tbody td::before {{
        content: attr(data-label);
        font-weight: 600;
        color: var(--ink-muted);
        text-align: left;
      }}
      .data-table tbody th {{
        font-family: var(--font-serif);
        font-size: 0.95rem;
        padding-block-end: 0.35rem;
        border-block-end: 1px solid var(--border-hairline);
        margin-block-end: 0.25rem;
      }}
    }}

    @media (prefers-reduced-motion: reduce) {{
      html {{
        scroll-behavior: auto;
      }}
      *, *::before, *::after {{
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
      }}
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to interactive film room</a>

  <header class="masthead">
    <div class="page-shell masthead-bar">
      <span class="brand-name">NFL BDB 2027</span>
      <nav class="top-nav" aria-label="Studio sections">
        <a href="#saliency-studio">Replay</a>
        <a href="#draft-ev">Calculator</a>
        <a href="#v4-evidence">Benchmarks</a>
        <a href="#figure-gallery">Figures</a>
      </nav>
    </div>
  </header>

  <main id="main-content" class="page-shell">
    <section class="editorial-header" aria-label="Studio overview">
      <div class="editorial-title-wrap">
        <h1>Optical Film-Room Studio</h1>
        <p class="editorial-deck">10 Hz Combine attention to NFL contract value.</p>
      </div>
      <div class="header-actions">
        <a class="btn btn-primary" href="#saliency-studio">Replay</a>
        <a class="hero-link" href="#draft-ev">Calculator</a>
      </div>
    </section>

    <!-- MODULE 1: Synchronized 10 Hz Ghost Comparison + 0.1s Dual-Head Temporal Attention Studio -->
    <section id="saliency-studio" class="studio-section" aria-labelledby="saliency-heading">
      <div class="section-head">
        <h2 id="saliency-heading">10 Hz Ghost Replay</h2>
        <p class="section-lead" id="matchupHeadline">{_esc(first["title"])}</p>
      </div>

      <div class="tablist-row" role="tablist" aria-label="Position group matchups">
          {tabs_markup}
      </div>

      <div class="matchup-brief" aria-live="polite">
        <div class="player-strip" id="gemBriefCard">
          <div class="player-strip-top">
            <span class="player-name-wrap">
              <span class="swatch" aria-hidden="true"></span>
              <span class="player-name" id="gemName">{_esc(first_gem["name"])}</span>
            </span>
            <span class="player-role">Gem</span>
          </div>
          <div class="player-meta num" id="gemMeta">
            {_esc(first_gem["pos"])}, Rd {int(first_gem["draft_label"].split()[1].rstrip(","))} #{first_gem["draft_label"].split("#")[-1]}, {_esc(first_gem["forty_label"])}, {first_gem["alpha"]:+.2f}σ ({first_gem["snaps"]:.0f} snaps)
          </div>
        </div>
        <div class="player-strip" id="trapBriefCard">
          <div class="player-strip-top">
            <span class="player-name-wrap">
              <span class="swatch swatch-trap" aria-hidden="true"></span>
              <span class="player-name" id="trapName">{_esc(first_trap["name"])}</span>
            </span>
            <span class="player-role player-role-trap">Peer</span>
          </div>
          <div class="player-meta num" id="trapMeta">
            {_esc(first_trap["pos"])}, {_esc(first_trap["draft_label"])}, {_esc(first_trap["forty_label"])}, {first_trap["alpha"]:+.2f}σ ({first_trap["snaps"]:.0f} snaps)
          </div>
        </div>
      </div>

      <div class="filmroom-grid">
        <div class="canvas-panel">
          <div class="canvas-head">
            <h3 class="canvas-title" id="fieldCanvasTitle">Ghost Turf Overlay ({_esc(first["drill_pretty"])})</h3>
            <span class="canvas-hint">Glow = attention</span>
          </div>
          <canvas id="fieldGhostCanvas" class="studio-canvas" width="600" height="390"
                  aria-label="Synchronized 10 Hz ghost trajectory comparison with attention saliency glow"></canvas>
        </div>

        <div class="canvas-panel">
          <div class="canvas-head">
            <h3 class="canvas-title">Attention Timeline</h3>
            <span class="canvas-hint">Hover to scrub</span>
          </div>
          <canvas id="saliencyCanvas" class="studio-canvas" width="600" height="390"
                  aria-label="0.1 second dual-head temporal attention and velocity timeline"></canvas>
        </div>
      </div>

      <div class="control-bar">
        <div class="transport-group">
          <button type="button" id="btnPlayPause" class="btn btn-primary">Play 10 Hz</button>
          <button type="button" id="btnStepBack" class="btn" aria-label="Step back 0.1 seconds">-0.1s</button>
          <button type="button" id="btnStepFwd" class="btn" aria-label="Step forward 0.1 seconds">+0.1s</button>
          <button type="button" id="btnJumpPeak" class="btn">Peak 0.3s</button>
        </div>

        <fieldset class="head-fieldset">
          <legend class="head-legend">Head</legend>
          <label class="radio-label"><input type="radio" name="attn-head" value="aw" checked><span>Combined α_t</span></label>
          <label class="radio-label"><input type="radio" name="attn-head" value="h1"><span>Burst</span></label>
          <label class="radio-label"><input type="radio" name="attn-head" value="h2"><span>Brake</span></label>
        </fieldset>

        <div class="scrubber-wrap">
          <label for="frameScrubber" class="scrubber-label">Frame</label>
          <input type="range" id="frameScrubber" min="0" max="{first_fmax}" step="1" value="{int(first_gem['peak_win']['peak_frame'])}" aria-valuetext="Peak frame">
          <output for="frameScrubber" id="frameReadout" class="scrubber-output num">t = 0.0s</output>
        </div>
      </div>

      <div class="telemetry-wrap">
        <table class="data-table" id="frameTelemetryTable">
          <caption id="telemetryCaption">0.1s Frame Telemetry</caption>
          <thead>
            <tr>
              <th scope="col">Player</th>
              <th scope="col" class="num">Time</th>
              <th scope="col" class="num">Speed</th>
              <th scope="col" class="num">Accel</th>
              <th scope="col" class="num">Cut</th>
              <th scope="col" class="num">α_t</th>
              <th scope="col" class="num">Burst</th>
              <th scope="col" class="num">Brake</th>
              <th scope="col" class="num">Peak 0.3s</th>
            </tr>
          </thead>
          <tbody id="telemetryBody">
            <tr>
              <th scope="row" id="telGemName" data-label="Player">{_esc(first_gem["name"])}</th>
              <td class="num" id="telGemT" data-label="Time">0.0s</td>
              <td class="num num-strong" id="telGemS" data-label="Speed">0.00</td>
              <td class="num" id="telGemA" data-label="Accel">0.00</td>
              <td class="num" id="telGemAc" data-label="Cut">0.00</td>
              <td class="num num-turf" id="telGemAw" data-label="α_t">0.000</td>
              <td class="num num-turf" id="telGemH1" data-label="Burst">0.000</td>
              <td class="num num-gold" id="telGemH2" data-label="Brake">0.000</td>
              <td class="num" id="telGemWin" data-label="Peak 0.3s">0.0 to 0.3s</td>
            </tr>
            <tr>
              <th scope="row" id="telTrapName" data-label="Player">{_esc(first_trap["name"])}</th>
              <td class="num" id="telTrapT" data-label="Time">0.0s</td>
              <td class="num num-strong" id="telTrapS" data-label="Speed">0.00</td>
              <td class="num" id="telTrapA" data-label="Accel">0.00</td>
              <td class="num" id="telTrapAc" data-label="Cut">0.00</td>
              <td class="num num-crimson" id="telTrapAw" data-label="α_t">0.000</td>
              <td class="num" id="telTrapH1" data-label="Burst">0.000</td>
              <td class="num" id="telTrapH2" data-label="Brake">0.000</td>
              <td class="num" id="telTrapWin" data-label="Peak 0.3s">0.0 to 0.3s</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- MODULE 2: Interactive GM Draft Room Surplus EV Calculator -->
    <section id="draft-ev" class="studio-section" aria-labelledby="ev-heading">
      <div class="section-head">
        <h2 id="ev-heading">GM Draft Surplus EV Calculator</h2>
        <p class="section-lead">Empirical 4-year rookie value calibrated on 510 prospects.</p>
      </div>

      <div class="ev-workspace">
        <form class="ev-controls-col" id="draftEvForm" onsubmit="return false;">
          <fieldset class="ev-fieldset">
            <legend>Position</legend>
            <div class="pos-radio-grid">
              <label class="pos-radio-item"><input type="radio" name="ev-pos-group" value="WR_TE" checked><span>Receivers</span></label>
              <label class="pos-radio-item"><input type="radio" name="ev-pos-group" value="DL_EDGE"><span>Edge / D-Line</span></label>
              <label class="pos-radio-item"><input type="radio" name="ev-pos-group" value="DB"><span>Defensive backs</span></label>
              <label class="pos-radio-item"><input type="radio" name="ev-pos-group" value="OL_LB"><span>Offensive line</span></label>
            </div>
          </fieldset>

          <div class="slider-field">
            <div class="slider-header">
              <label for="draftPickSlider">Draft pick</label>
              <output for="draftPickSlider" id="draftPickOut" class="num">#177, Rd 5 ($4.21M)</output>
            </div>
            <input type="range" id="draftPickSlider" min="1" max="257" step="1" value="177" aria-valuetext="Pick 177, Round 5">
            <div class="slider-scale num">
              <span>#1</span>
              <span>Rd 2</span>
              <span>Rd 5</span>
              <span>#257</span>
            </div>
          </div>

          <div class="slider-field">
            <div class="slider-header">
              <label for="opticalAlphaSlider">Optical α (σ)</label>
              <output for="opticalAlphaSlider" id="opticalAlphaOut" class="num">+0.6σ</output>
            </div>
            <input type="range" id="opticalAlphaSlider" min="-2.5" max="2.5" step="0.1" value="0.6" aria-valuetext="+0.6 standard deviations">
            <div class="slider-scale num">
              <span>-2.5σ</span>
              <span>0.0σ</span>
              <span>+2.5σ</span>
            </div>
          </div>

          <div class="preset-stack">
            <p class="preset-heading">Presets</p>
            <button type="button" class="btn btn-preset" data-preset-grp="WR_TE" data-preset-pick="177" data-preset-oz="0.6">
              Puka Nacua (#177, +0.6σ)
            </button>
            <button type="button" class="btn btn-preset" data-preset-grp="DL_EDGE" data-preset-pick="54" data-preset-oz="0.5">
              Tuli Tuipulotu (#54, +0.5σ)
            </button>
            <button type="button" class="btn btn-preset" data-preset-grp="DB" data-preset-pick="24" data-preset-oz="-1.4">
              DB Trap (#24, -1.4σ)
            </button>
          </div>
        </form>

        <div class="ev-output-col">
          <div class="ev-kpi-row" aria-live="polite">
            <div class="ev-kpi" id="evKpiSurplusCard">
              <span class="ev-kpi-label">4-yr surplus EV</span>
              <span class="ev-kpi-val num" id="evSurplusVal">+$0.0M</span>
              <span class="ev-kpi-sub num" id="evSurplusCi">CI: [$0.0M, $0.0M]</span>
            </div>
            <div class="ev-kpi ev-kpi-gold">
              <span class="ev-kpi-label">Expected value</span>
              <span class="ev-kpi-val num" id="evProdVal">$0.0M</span>
              <span class="ev-kpi-sub num" id="evCostSub">Cap: $0.0M</span>
            </div>
            <div class="ev-kpi" id="evKpiBustCard">
              <span class="ev-kpi-label">Bust risk delta</span>
              <span class="ev-kpi-val num" id="evBustRedVal">0.0%</span>
              <span class="ev-kpi-sub num" id="evBustProbSub">0% to 0%</span>
            </div>
          </div>

          <div class="verdict-strip" id="evVerdictStrip">
            <span class="verdict-label-wrap">
              <span class="swatch" id="evVerdictSwatch" aria-hidden="true"></span>
              <strong id="evVerdictText">Verdict</strong>
            </span>
            <span class="num" id="evConservationCheck">$0M - $0M = $0M</span>
          </div>

          <div class="canvas-panel">
            <div class="canvas-head">
              <h3 class="canvas-title" id="evChartTitle">Surplus EV Curve</h3>
              <span class="canvas-hint">Band = 95% CI</span>
            </div>
            <canvas id="draftEvCanvas" class="studio-canvas" width="720" height="330"
                    aria-label="Draft round surplus expected value curve and historical prospects"></canvas>
          </div>

          <div>
            <h3 class="canvas-title" id="compsHeading">Real Historical Comps</h3>
            <div class="comps-grid" id="historicalCompsGrid" aria-live="polite"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- MODULE 3: V4 Statistical Rigor & Benchmark Evidence -->
    <section id="v4-evidence" class="studio-section" aria-labelledby="rigor-heading">
      <div class="section-head">
        <h2 id="rigor-heading">V4 Positional Benchmarks</h2>
        <p class="section-lead">
          OOF R² +{ov["cv_r2"]:.4f} (LOYO +{ov["temporal_loyo_r2"]:.4f}), DL/EDGE +{pos_enc["DL_EDGE"]["cv_r2"]:.4f} (LOYO +{pos_enc["DL_EDGE"]["temporal_loyo_r2"]:.4f}), Partial R² +{ov["partial_r2_vs_draft_stopwatch"]:.4f} (q={ov["q_fdr_vs_draft_stopwatch"]:.4f}).
        </p>
      </div>

      <div class="table-scroll">
        <table class="data-table">
          <caption>Nested CV R² Across 4 Position Encoders</caption>
          <thead>
            <tr>
              <th scope="col">Encoder</th>
              <th scope="col" class="num">Cohort</th>
              <th scope="col" class="num">Stopwatch</th>
              <th scope="col" class="num">V2</th>
              <th scope="col" class="num">V4 OOF</th>
              <th scope="col" class="num">LOYO</th>
              <th scope="col" class="num">Partial R²</th>
              <th scope="col" class="num">q_FDR</th>
            </tr>
          </thead>
          <tbody>
            {rigor_tbody}
          </tbody>
        </table>
      </div>
    </section>

    <!-- MODULE 4: 6 Economist/NYT 2x Retina Publication Figures -->
    <section id="figure-gallery" class="studio-section" aria-labelledby="gallery-heading">
      <div class="section-head">
        <h2 id="gallery-heading">Publication Figures</h2>
      </div>

      <div class="gallery-grid">
        {gallery_markup}
      </div>
    </section>
  </main>

  <footer class="page-shell site-footer">
    <span>NFL Big Data Bowl 2027</span>
    <span class="num">431,094 Combine frames, 22.65M NFL frames, 510 rookies</span>
  </footer>

  <script id="v5-data" type="application/json">{json_blob}</script>
  <script>
  (() => {{
    const DATA = JSON.parse(document.getElementById("v5-data").textContent);
    const MATCHUPS = DATA.matchups;
    const EV = DATA.ev_bundle;

    const COLORS = {{
      paper: "#FFFFFF",
      subtle: "#F7F7F7",
      tint: "#F2F2F2",
      rule: "#E2E2E2",
      ink: "#111111",
      secondary: "#4A4A4A",
      muted: "#666666",
      green: "#184E32",
      greenMid: "#2E6F4E",
      greenTint: "#EAF2EE",
      crimson: "#C43C29",
      crimsonTint: "#FCEEEE",
      gold: "#C68B2C",
      goldInk: "#875E12",
      goldTint: "#FBF4E4"
    }};

    const REDUCED_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function syncRangeVisual(el) {{
      el.style.setProperty("--attr-min", el.min || "0");
      el.style.setProperty("--attr-max", el.max || "100");
      el.style.setProperty("--control-value", el.value);
    }}

    const state = {{
      matchupIdx: 0,
      frameIdx: MATCHUPS[0].gem.peak_win.peak_frame,
      headKey: "aw",
      playing: false,
      timerId: null
    }};

    const fieldCanvas = document.getElementById("fieldGhostCanvas");
    const salCanvas = document.getElementById("saliencyCanvas");
    const frameScrubber = document.getElementById("frameScrubber");
    const frameReadout = document.getElementById("frameReadout");
    const btnPlayPause = document.getElementById("btnPlayPause");
    const tabs = Array.from(document.querySelectorAll(".matchup-tab"));

    function currentMatchup() {{
      return MATCHUPS[state.matchupIdx];
    }}

    function maxFramesFor(m) {{
      return Math.max(m.gem.frames.length, m.trap.frames.length);
    }}

    function clampFrame(arr, idx) {{
      return arr[Math.max(0, Math.min(arr.length - 1, idx))];
    }}

    function setupHiDPI(canvas, logicalW, logicalH) {{
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const cssW = Math.max(260, Math.round(rect.width || logicalW));
      const cssH = Math.round(cssW * (logicalH / logicalW));
      if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {{
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
      }}
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, w: cssW, h: cssH }};
    }}

    function getHeadLabel(k) {{
      if (k === "h1") return "Head 1 Burst";
      if (k === "h2") return "Head 2 Brake";
      return "Combined α_t";
    }}

    function updateMatchupMetaUI() {{
      const m = currentMatchup();
      document.getElementById("matchupHeadline").textContent = m.title;
      document.getElementById("fieldCanvasTitle").textContent =
        `Ghost Turf Overlay (${{m.drill_pretty}})`;
      document.getElementById("gemName").textContent = m.gem.name;
      document.getElementById("gemMeta").textContent =
        `${{m.gem.pos}}, ${{m.gem.draft_label}}, ${{m.gem.forty_label}}, ${{m.gem.alpha >= 0 ? "+" : ""}}${{m.gem.alpha.toFixed(2)}}σ (${{m.gem.snaps}} snaps)`;
      document.getElementById("trapName").textContent = m.trap.name;
      document.getElementById("trapMeta").textContent =
        `${{m.trap.pos}}, ${{m.trap.draft_label}}, ${{m.trap.forty_label}}, ${{m.trap.alpha >= 0 ? "+" : ""}}${{m.trap.alpha.toFixed(2)}}σ (${{m.trap.snaps}} snaps)`;

      const fMax = maxFramesFor(m) - 1;
      frameScrubber.max = String(fMax);
      frameScrubber.value = String(Math.min(state.frameIdx, fMax));
      syncRangeVisual(frameScrubber);
    }}

    function updateTelemetryTable() {{
      const m = currentMatchup();
      const fg = clampFrame(m.gem.frames, state.frameIdx);
      const ft = clampFrame(m.trap.frames, state.frameIdx);

      frameScrubber.value = String(state.frameIdx);
      frameScrubber.setAttribute("aria-valuetext", `Frame ${{state.frameIdx}}, time ${{fg.t.toFixed(1)}} seconds`);
      syncRangeVisual(frameScrubber);
      frameReadout.textContent = `t = ${{fg.t.toFixed(1)}}s (f${{state.frameIdx}})`;

      document.getElementById("telGemName").textContent = m.gem.name;
      document.getElementById("telGemT").textContent = `${{fg.t.toFixed(1)}}s`;
      document.getElementById("telGemS").textContent = fg.s.toFixed(2);
      document.getElementById("telGemA").textContent = `${{fg.a >= 0 ? "+" : ""}}${{fg.a.toFixed(2)}}`;
      document.getElementById("telGemAc").textContent = fg.ac.toFixed(2);
      document.getElementById("telGemAw").textContent = fg.aw.toFixed(4);
      document.getElementById("telGemH1").textContent = fg.h1.toFixed(4);
      document.getElementById("telGemH2").textContent = fg.h2.toFixed(4);
      document.getElementById("telGemWin").textContent =
        `${{m.gem.peak_win.start_sec.toFixed(1)}} to ${{m.gem.peak_win.end_sec.toFixed(1)}}s`;

      document.getElementById("telTrapName").textContent = m.trap.name;
      document.getElementById("telTrapT").textContent = `${{ft.t.toFixed(1)}}s`;
      document.getElementById("telTrapS").textContent = ft.s.toFixed(2);
      document.getElementById("telTrapA").textContent = `${{ft.a >= 0 ? "+" : ""}}${{ft.a.toFixed(2)}}`;
      document.getElementById("telTrapAc").textContent = ft.ac.toFixed(2);
      document.getElementById("telTrapAw").textContent = ft.aw.toFixed(4);
      document.getElementById("telTrapH1").textContent = ft.h1.toFixed(4);
      document.getElementById("telTrapH2").textContent = ft.h2.toFixed(4);
      document.getElementById("telTrapWin").textContent =
        `${{m.trap.peak_win.start_sec.toFixed(1)}} to ${{m.trap.peak_win.end_sec.toFixed(1)}}s`;
    }}

    function drawRadialGlow(ctx, cx, cy, r, hexColor, peakAlpha) {{
      ctx.save();
      ctx.globalAlpha = peakAlpha * 0.45;
      ctx.fillStyle = hexColor;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = peakAlpha * 0.72;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.56, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }}

    function drawFieldGhost() {{
      const m = currentMatchup();
      const {{ ctx, w, h }} = setupHiDPI(fieldCanvas, 600, 390);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = COLORS.paper;
      ctx.fillRect(0, 0, w, h);

      const pad = {{ l: 46, r: 20, t: 26, b: 34 }};
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      const allPts = [...m.gem.frames, ...m.trap.frames];
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of allPts) {{
        if (p.dx < minX) minX = p.dx;
        if (p.dx > maxX) maxX = p.dx;
        if (p.dy < minY) minY = p.dy;
        if (p.dy > maxY) maxY = p.dy;
      }}
      const spanX = Math.max(4.5, maxX - minX);
      const spanY = Math.max(4.5, maxY - minY);
      const x0 = minX - spanX * 0.14, x1 = maxX + spanX * 0.14;
      const y0 = minY - spanY * 0.16, y1 = maxY + spanY * 0.16;

      const mapX = (dx) => pad.l + ((dx - x0) / (x1 - x0)) * pw;
      const mapY = (dy) => pad.t + ph - ((dy - y0) / (y1 - y0)) * ph;

      ctx.fillStyle = COLORS.subtle;
      ctx.fillRect(pad.l, pad.t, pw, ph);
      ctx.strokeStyle = COLORS.rule;
      ctx.lineWidth = 1;

      const stepX = (x1 - x0) > 20 ? 5 : 2;
      const stepY = (y1 - y0) > 20 ? 5 : 2;
      ctx.font = "11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillStyle = COLORS.muted;

      for (let gx = Math.ceil(x0 / stepX) * stepX; gx <= x1; gx += stepX) {{
        const px = mapX(gx);
        ctx.beginPath();
        ctx.moveTo(px, pad.t);
        ctx.lineTo(px, pad.t + ph);
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(`${{gx >= 0 ? "+" : ""}}${{gx}}yd`, px, pad.t + ph + 16);
      }}
      for (let gy = Math.ceil(y0 / stepY) * stepY; gy <= y1; gy += stepY) {{
        const py = mapY(gy);
        ctx.beginPath();
        ctx.moveTo(pad.l, py);
        ctx.lineTo(pad.l + pw, py);
        ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(`${{gy >= 0 ? "+" : ""}}${{gy}}`, pad.l - 6, py + 4);
      }}

      const ox = mapX(0);
      ctx.strokeStyle = COLORS.ink;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(ox, pad.t);
      ctx.lineTo(ox, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);

      const hk = state.headKey;
      const maxW = Math.max(
        0.012,
        ...m.gem.frames.map(f => f[hk]),
        ...m.trap.frames.map(f => f[hk])
      );

      function drawGhostPath(player, colorHex, glowHex, dashed) {{
        const frs = player.frames;
        const limit = Math.min(frs.length - 1, state.frameIdx);

        ctx.save();
        ctx.strokeStyle = colorHex;
        ctx.globalAlpha = 0.24;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([5, 3]);
        ctx.beginPath();
        frs.forEach((f, i) => {{
          const px = mapX(f.dx), py = mapY(f.dy);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }});
        ctx.stroke();
        ctx.restore();

        const pwStart = player.peak_win.start_frame;
        const pwEnd = player.peak_win.end_frame;
        for (let i = pwStart; i <= Math.min(pwEnd, frs.length - 1); i++) {{
          const pf = frs[i];
          drawRadialGlow(ctx, mapX(pf.dx), mapY(pf.dy), 15, glowHex, 0.55);
        }}

        ctx.save();
        ctx.strokeStyle = colorHex;
        ctx.lineWidth = dashed ? 2.4 : 3.2;
        if (dashed) ctx.setLineDash([6, 3]);
        ctx.beginPath();
        for (let i = 0; i <= limit; i++) {{
          const px = mapX(frs[i].dx), py = mapY(frs[i].dy);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }}
        ctx.stroke();
        ctx.restore();

        for (let i = 0; i <= limit; i++) {{
          const f = frs[i];
          const norm = Math.min(1, f[hk] / maxW);
          if (norm > 0.42) {{
            drawRadialGlow(ctx, mapX(f.dx), mapY(f.dy), 5 + norm * 11, glowHex, 0.28 + norm * 0.34);
          }}
        }}

        const cur = frs[limit];
        const cx = mapX(cur.dx), cy = mapY(cur.dy);
        const curNorm = Math.min(1, cur[hk] / maxW);
        drawRadialGlow(ctx, cx, cy, 11 + curNorm * 16, glowHex, 0.45 + curNorm * 0.4);

        ctx.fillStyle = colorHex;
        ctx.strokeStyle = COLORS.paper;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, 6.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }}

      const gemGlowHex = hk === "h2" ? COLORS.gold : COLORS.green;
      drawGhostPath(m.trap, COLORS.crimson, COLORS.crimson, true);
      drawGhostPath(m.gem, COLORS.green, gemGlowHex, false);

      // Pick least-crowded corner (bottom-left vs top-left) so legend never overlaps glowing trajectories
      const legW = 220, legH = 46;
      const candCorners = [
        {{ x: pad.l + 8, y: pad.t + ph - legH - 8 }},
        {{ x: pad.l + 8, y: pad.t + 8 }},
        {{ x: pad.l + pw - legW - 8, y: pad.t + 8 }},
        {{ x: pad.l + pw - legW - 8, y: pad.t + ph - legH - 8 }}
      ];
      let bestCorner = candCorners[0];
      let bestHits = Infinity;
      for (const c of candCorners) {{
        let hits = 0;
        for (const p of allPts) {{
          const px = mapX(p.dx), py = mapY(p.dy);
          if (px >= c.x - 14 && px <= c.x + legW + 14 && py >= c.y - 14 && py <= c.y + legH + 14) hits++;
        }}
        if (hits < bestHits) {{ bestHits = hits; bestCorner = c; }}
      }}
      const lx = bestCorner.x, ly = bestCorner.y;

      ctx.save();
      ctx.globalAlpha = 0.94;
      ctx.fillStyle = COLORS.paper;
      ctx.fillRect(lx, ly, legW, legH);
      ctx.restore();
      ctx.strokeStyle = COLORS.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(lx, ly, legW, legH);

      ctx.strokeStyle = COLORS.green;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(lx + 8, ly + 15);
      ctx.lineTo(lx + 28, ly + 15);
      ctx.stroke();
      ctx.fillStyle = COLORS.ink;
      ctx.textAlign = "left";
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillText(`${{m.gem.name}} (${{getHeadLabel(hk)}})`, lx + 34, ly + 19);

      ctx.strokeStyle = COLORS.crimson;
      ctx.lineWidth = 2.2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(lx + 8, ly + 34);
      ctx.lineTo(lx + 28, ly + 34);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.crimson;
      ctx.fillText(`${{m.trap.name}} (Peer)`, lx + 34, ly + 38);

      ctx.strokeStyle = COLORS.rule;
      ctx.strokeRect(pad.l, pad.t, pw, ph);
    }}

    function drawSaliencyTimeline() {{
      const m = currentMatchup();
      const {{ ctx, w, h }} = setupHiDPI(salCanvas, 600, 390);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = COLORS.paper;
      ctx.fillRect(0, 0, w, h);

      const pad = {{ l: 48, r: 18, t: 24, b: 32 }};
      const pw = w - pad.l - pad.r;
      const topH = Math.round((h - pad.t - pad.b - 22) * 0.46);
      const botY = pad.t + topH + 22;
      const botH = h - botY - pad.b;

      const fCount = maxFramesFor(m);
      const tMax = Math.max(0.5, (fCount - 1) * 0.1);
      const mapT = (t) => pad.l + (t / tMax) * pw;

      const maxS = Math.max(
        8.0,
        ...m.gem.frames.map(f => f.s),
        ...m.trap.frames.map(f => f.s)
      ) * 1.1;
      const mapS = (s) => pad.t + topH - (s / maxS) * topH;

      const hk = state.headKey;
      const maxA = Math.max(
        0.025,
        ...m.gem.frames.map(f => Math.max(f.aw, f.h1, f.h2)),
        ...m.trap.frames.map(f => Math.max(f.aw, f.h1, f.h2))
      ) * 1.15;
      const mapA = (v) => botY + botH - (v / maxA) * botH;

      const pk = m.gem.peak_win;
      const wx1 = mapT(pk.start_sec);
      const wx2 = Math.max(wx1 + 10, mapT(pk.end_sec));
      ctx.fillStyle = COLORS.crimsonTint;
      ctx.fillRect(wx1, pad.t, wx2 - wx1, topH);
      ctx.fillRect(wx1, botY, wx2 - wx1, botH);

      ctx.strokeStyle = COLORS.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.l, pad.t, pw, topH);
      ctx.font = "11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "right";
      for (const st of [0, Math.round(maxS / 2), Math.round(maxS)]) {{
        const y = mapS(st);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.fillText(`${{st}}`, pad.l - 6, y + 4);
      }}
      ctx.textAlign = "left";
      ctx.fillStyle = COLORS.ink;
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillText("Optical Speed (yd/s)", pad.l + 6, pad.t + 14);
      ctx.fillStyle = COLORS.crimson;
      ctx.textAlign = "right";
      ctx.fillText(`Peak 0.3s: ${{pk.start_sec.toFixed(1)}} to ${{pk.end_sec.toFixed(1)}}s`, pad.l + pw - 6, pad.t + 14);

      function plotLine(frames, getter, yFn, color, width, dashed) {{
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        if (dashed) ctx.setLineDash([5, 3]);
        ctx.beginPath();
        frames.forEach((f, i) => {{
          const x = mapT(f.t), y = yFn(getter(f));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
        ctx.restore();
      }}

      plotLine(m.trap.frames, f => f.s, mapS, COLORS.crimson, 2.0, true);
      plotLine(m.gem.frames, f => f.s, mapS, COLORS.green, 2.6, false);

      ctx.strokeStyle = COLORS.rule;
      ctx.strokeRect(pad.l, botY, pw, botH);
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "right";
      ctx.font = "11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillText("0.00", pad.l - 6, botY + botH);
      ctx.fillText(maxA.toFixed(2), pad.l - 6, botY + 10);

      ctx.save();
      ctx.globalAlpha = 0.20;
      ctx.fillStyle = hk === "h2" ? COLORS.gold : COLORS.green;
      ctx.beginPath();
      ctx.moveTo(mapT(m.gem.frames[0].t), botY + botH);
      m.gem.frames.forEach(f => ctx.lineTo(mapT(f.t), mapA(f[hk])));
      ctx.lineTo(mapT(m.gem.frames[m.gem.frames.length - 1].t), botY + botH);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      plotLine(m.gem.frames, f => f.h1, mapA, COLORS.green, hk === "h1" ? 2.5 : 1.4, false);
      plotLine(m.gem.frames, f => f.h2, mapA, COLORS.goldInk, hk === "h2" ? 2.5 : 1.4, true);
      plotLine(m.trap.frames, f => f[hk], mapA, COLORS.crimson, 1.8, true);

      ctx.textAlign = "left";
      ctx.fillStyle = COLORS.ink;
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.fillText(`0.1s Attention (${{getHeadLabel(hk)}})`, pad.l + 6, botY + 14);

      ctx.fillStyle = COLORS.muted;
      ctx.font = "11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      ctx.textAlign = "center";
      const tStep = tMax > 6 ? 2 : 1;
      for (let t = 0; t <= tMax + 0.01; t += tStep) {{
        ctx.fillText(`${{t.toFixed(0)}}.0s`, mapT(t), botY + botH + 18);
      }}

      const curT = state.frameIdx * 0.1;
      const curX = mapT(curT);
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(curX, pad.t);
      ctx.lineTo(curX, botY + botH);
      ctx.stroke();

      const fg = clampFrame(m.gem.frames, state.frameIdx);
      ctx.fillStyle = COLORS.green;
      ctx.beginPath();
      ctx.arc(curX, mapS(fg.s), 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(curX, mapA(fg[hk]), 4.5, 0, Math.PI * 2);
      ctx.fill();
    }}

    function renderSaliencyAll() {{
      updateTelemetryTable();
      drawFieldGhost();
      drawSaliencyTimeline();
    }}

    function stopPlayback() {{
      state.playing = false;
      if (state.timerId !== null) {{
        clearInterval(state.timerId);
        state.timerId = null;
      }}
      btnPlayPause.textContent = "Play 10 Hz";
    }}

    function startPlayback() {{
      stopPlayback();
      state.playing = true;
      btnPlayPause.textContent = "Pause 10 Hz";
      state.timerId = setInterval(() => {{
        const maxF = maxFramesFor(currentMatchup()) - 1;
        state.frameIdx = state.frameIdx >= maxF ? 0 : state.frameIdx + 1;
        renderSaliencyAll();
      }}, 100);
    }}

    function selectMatchup(idx, focusTab) {{
      stopPlayback();
      state.matchupIdx = idx;
      state.frameIdx = MATCHUPS[idx].gem.peak_win.peak_frame;
      tabs.forEach((tab, i) => {{
        const active = i === idx;
        tab.setAttribute("aria-selected", String(active));
        tab.tabIndex = active ? 0 : -1;
      }});
      if (focusTab) tabs[idx].focus();
      tabs[idx].scrollIntoView({{ block: "nearest", inline: "nearest" }});
      updateMatchupMetaUI();
      renderSaliencyAll();
    }}

    tabs.forEach((tab, i) => {{
      tab.addEventListener("click", () => selectMatchup(i, false));
      tab.addEventListener("keydown", (e) => {{
        let next = null;
        if (e.key === "ArrowRight") next = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") next = 0;
        else if (e.key === "End") next = tabs.length - 1;
        if (next !== null) {{
          e.preventDefault();
          selectMatchup(next, true);
        }}
      }});
    }});

    btnPlayPause.addEventListener("click", () => {{
      if (state.playing) stopPlayback(); else startPlayback();
    }});
    document.getElementById("btnStepBack").addEventListener("click", () => {{
      stopPlayback();
      state.frameIdx = Math.max(0, state.frameIdx - 1);
      renderSaliencyAll();
    }});
    document.getElementById("btnStepFwd").addEventListener("click", () => {{
      stopPlayback();
      const maxF = maxFramesFor(currentMatchup()) - 1;
      state.frameIdx = Math.min(maxF, state.frameIdx + 1);
      renderSaliencyAll();
    }});
    document.getElementById("btnJumpPeak").addEventListener("click", () => {{
      stopPlayback();
      state.frameIdx = currentMatchup().gem.peak_win.peak_frame;
      renderSaliencyAll();
    }});

    frameScrubber.addEventListener("input", () => {{
      stopPlayback();
      state.frameIdx = parseInt(frameScrubber.value, 10) || 0;
      renderSaliencyAll();
    }});

    document.querySelectorAll('input[name="attn-head"]').forEach((r) => {{
      r.addEventListener("change", () => {{
        if (r.checked) {{
          state.headKey = r.value;
          renderSaliencyAll();
        }}
      }});
    }});

    function scrubFromPointer(evt) {{
      const rect = salCanvas.getBoundingClientRect();
      const padL = 48, padR = 18;
      const relX = evt.clientX - rect.left - padL;
      const span = Math.max(1, rect.width - padL - padR);
      const frac = Math.max(0, Math.min(1, relX / span));
      const maxF = maxFramesFor(currentMatchup()) - 1;
      const newIdx = Math.round(frac * maxF);
      if (newIdx !== state.frameIdx) {{
        stopPlayback();
        state.frameIdx = newIdx;
        renderSaliencyAll();
      }}
    }}
    salCanvas.addEventListener("pointermove", scrubFromPointer);
    salCanvas.addEventListener("click", scrubFromPointer);

    /* =====================================================================
       MODULE 2: Interactive GM Draft-Room Surplus EV Calculator
       ===================================================================== */
    const evCanvas = document.getElementById("draftEvCanvas");
    const pickSlider = document.getElementById("draftPickSlider");
    const optSlider = document.getElementById("opticalAlphaSlider");
    const evState = {{
      grp: "WR_TE",
      pick: 177,
      oz: 0.6
    }};

    const CBA_LOG_X = EV.cba_pick_anchors.map(p => Math.log(p));
    const CBA_Y = EV.cba_cost_4yr_m_anchors;

    function rookieCost4Yr(pick) {{
      const lp = Math.log(Math.max(1, Math.min(257, pick)));
      for (let i = 0; i < CBA_LOG_X.length - 1; i++) {{
        if (lp >= CBA_LOG_X[i] && lp <= CBA_LOG_X[i + 1]) {{
          const t = (lp - CBA_LOG_X[i]) / (CBA_LOG_X[i + 1] - CBA_LOG_X[i]);
          return +(CBA_Y[i] + t * (CBA_Y[i + 1] - CBA_Y[i])).toFixed(3);
        }}
      }}
      return CBA_Y[CBA_Y.length - 1];
    }}

    function pickToRound(p) {{
      const bounds = [32, 64, 100, 135, 177, 220, 257];
      for (let i = 0; i < bounds.length; i++) {{
        if (p <= bounds[i]) return i + 1;
      }}
      return 7;
    }}

    function sigmoid(z) {{
      return 1 / (1 + Math.exp(-z));
    }}

    function evalDraftScenario(grp, pick, oz) {{
      const pm = EV.model_params[grp];
      const lp = Math.log(pick);
      const cost = rookieCost4Yr(pick);
      const prod = +Math.max(2.4, pm.intercept + pm.b_log_pick * lp + pm.b_optical_z * oz).toFixed(3);
      const surplus = +(prod - cost).toFixed(3);
      const halfW = +(1.96 * pm.resid_se_m * Math.sqrt(1 + 0.12 * oz * oz)).toFixed(3);
      const pSw = sigmoid(pm.bust_sw_b0 + pm.bust_sw_b1 * lp);
      const pOpt = sigmoid(pm.bust_opt_b0 + pm.bust_opt_b1 * lp + pm.bust_opt_b2 * oz);
      const bustRed = +(100 * (pSw - pOpt) / Math.max(pSw, 1e-4)).toFixed(1);

      let verdict = "Slot-neutral market pick";
      let isRisk = false;
      if (surplus >= 12.0 && oz >= 0.75) verdict = "High-conviction optical value target";
      else if (surplus >= 5.0 && oz >= 0.0) verdict = "Positive surplus value pick";
      else if (oz <= -0.75 || surplus < -1.5) {{ verdict = "Elevated stopwatch trap risk"; isRisk = true; }}

      const pool = EV.prospects.filter(r => r.grp === grp);
      const scored = pool.map(r => {{
        const dp = (Math.log(Math.max(1, r.pk)) - lp) / 1.15;
        const dz = (r.oz - oz) / 1.0;
        const d = Math.sqrt(dp * dp + dz * dz);
        return {{ ...r, sim: 1 / (1 + d), dist: d }};
      }}).sort((a, b) => a.dist - b.dist || a.id - b.id);

      return {{
        grp,
        pick,
        round: pickToRound(pick),
        oz,
        cost,
        prod,
        surplus,
        ciLo: +(surplus - halfW).toFixed(2),
        ciHi: +(surplus + halfW).toFixed(2),
        pSw,
        pOpt,
        bustRed,
        verdict,
        isRisk,
        comps: scored.slice(0, 3),
        pool
      }};
    }}

    function drawDraftEvChart(res) {{
      const {{ ctx, w, h }} = setupHiDPI(evCanvas, 720, 330);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = COLORS.paper;
      ctx.fillRect(0, 0, w, h);

      const pad = {{ l: 58, r: 22, t: 20, b: 36 }};
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;

      const yMin = -24, yMax = 56;
      const mapX = (pk) => pad.l + ((pk - 1) / 256) * pw;
      const mapY = (v) => pad.t + ph - ((Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * ph;

      ctx.font = "11px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
      for (const yt of [-20, -10, 0, 10, 20, 30, 40, 50]) {{
        const y = mapY(yt);
        ctx.strokeStyle = yt === 0 ? COLORS.ink : COLORS.rule;
        ctx.lineWidth = yt === 0 ? 1.2 : 0.8;
        ctx.setLineDash(yt === 0 ? [5, 3] : []);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + pw, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.muted;
        ctx.textAlign = "right";
        ctx.fillText(yt === 0 ? "$0M" : (yt > 0 ? `+$${{yt}}M` : `-$${{Math.abs(yt)}}M`), pad.l - 7, y + 4);
      }}

      const rdTicks = [[1, "#1"], [32, "Rd1"], [64, "Rd2"], [100, "Rd3"], [135, "Rd4"], [177, "Rd5"], [220, "Rd6"], [257, "Rd7"]];
      for (const [pk, lbl] of rdTicks) {{
        const x = mapX(pk);
        ctx.strokeStyle = COLORS.tint;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, pad.t);
        ctx.lineTo(x, pad.t + ph);
        ctx.stroke();
        ctx.fillStyle = COLORS.muted;
        ctx.textAlign = "center";
        ctx.fillText(lbl, x, pad.t + ph + 18);
      }}

      ctx.save();
      ctx.globalAlpha = 0.22;
      for (const p of res.pool) {{
        ctx.fillStyle = p.oz >= 0.5 ? COLORS.green : (p.oz <= -0.5 ? COLORS.crimson : COLORS.muted);
        ctx.beginPath();
        ctx.arc(mapX(p.pk), mapY(p.surplus), 3.1, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.restore();

      const samplePicks = [1, 4, 8, 14, 22, 32, 48, 64, 84, 108, 135, 165, 195, 225, 257];
      const liveCurve = samplePicks.map(pk => evalDraftScenario(res.grp, pk, res.oz));
      const medCurve = EV.curves_by_group[res.grp].median_opt_00;

      ctx.save();
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = res.oz >= 0 ? COLORS.green : COLORS.crimson;
      ctx.beginPath();
      liveCurve.forEach((pt, i) => {{
        const x = mapX(pt.pick), y = mapY(pt.ciHi);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      for (let i = liveCurve.length - 1; i >= 0; i--) {{
        ctx.lineTo(mapX(liveCurve[i].pick), mapY(liveCurve[i].ciLo));
      }}
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      ctx.save();
      ctx.strokeStyle = COLORS.muted;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      medCurve.forEach((pt, i) => {{
        const x = mapX(pt.pick), y = mapY(pt.surplus_m);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.restore();

      const activeCol = res.oz >= 0.4 ? COLORS.green : (res.oz <= -0.4 ? COLORS.crimson : COLORS.goldInk);
      ctx.strokeStyle = activeCol;
      ctx.lineWidth = 3;
      ctx.beginPath();
      liveCurve.forEach((pt, i) => {{
        const x = mapX(pt.pick), y = mapY(pt.surplus);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      res.comps.forEach((c, idx) => {{
        const cx = mapX(c.pk), cy = mapY(c.surplus);
        ctx.fillStyle = COLORS.goldTint;
        ctx.strokeStyle = COLORS.goldInk;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(cx, cy, 8.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = COLORS.ink;
        ctx.font = "700 10px -apple-system, BlinkMacSystemFont, Helvetica Neue, Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(String(idx + 1), cx, cy + 3.5);
      }});

      const ax = mapX(res.pick), ay = mapY(res.surplus);
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(ax, pad.t);
      ctx.lineTo(ax, pad.t + ph);
      ctx.moveTo(pad.l, ay);
      ctx.lineTo(pad.l + pw, ay);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = activeCol;
      ctx.strokeStyle = COLORS.paper;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(ax, ay, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.strokeStyle = COLORS.rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.l, pad.t, pw, ph);
    }}

    function renderDraftEvAll() {{
      syncRangeVisual(pickSlider);
      syncRangeVisual(optSlider);
      const res = evalDraftScenario(evState.grp, evState.pick, evState.oz);

      const pickText = `#${{res.pick}}, Rd ${{res.round}} ($${{res.cost.toFixed(2)}}M)`;
      document.getElementById("draftPickOut").textContent = pickText;
      pickSlider.setAttribute("aria-valuetext", pickText);

      const ozText = `${{res.oz >= 0 ? "+" : ""}}${{res.oz.toFixed(1)}}σ`;
      document.getElementById("opticalAlphaOut").textContent = ozText;
      optSlider.setAttribute("aria-valuetext", `${{ozText}} optical kinematic alpha`);

      const surSign = res.surplus >= 0 ? "+" : "-";
      const surEl = document.getElementById("evSurplusVal");
      surEl.textContent = `${{surSign}}$${{Math.abs(res.surplus).toFixed(1)}}M`;
      surEl.className = `ev-kpi-val num ${{res.surplus >= 0 ? "num-turf" : "num-crimson"}}`;
      document.getElementById("evKpiSurplusCard").className =
        res.surplus >= 0 ? "ev-kpi" : "ev-kpi ev-kpi-crimson";
      document.getElementById("evSurplusCi").textContent =
        `CI: [${{res.ciLo >= 0 ? "+" : ""}}$${{res.ciLo.toFixed(1)}}M, ${{res.ciHi >= 0 ? "+" : ""}}$${{res.ciHi.toFixed(1)}}M]`;

      document.getElementById("evProdVal").textContent = `$${{res.prod.toFixed(1)}}M`;
      document.getElementById("evCostSub").textContent = `Cap: $${{res.cost.toFixed(2)}}M`;

      const brSign = res.bustRed >= 0 ? "-" : "+";
      const brEl = document.getElementById("evBustRedVal");
      brEl.textContent = `${{brSign}}${{Math.abs(res.bustRed).toFixed(1)}}%`;
      brEl.className = `ev-kpi-val num ${{res.bustRed >= 0 ? "num-turf" : "num-crimson"}}`;
      document.getElementById("evKpiBustCard").className =
        res.bustRed >= 0 ? "ev-kpi" : "ev-kpi ev-kpi-crimson";
      document.getElementById("evBustProbSub").textContent =
        `${{(res.pSw * 100).toFixed(0)}}% to ${{(res.pOpt * 100).toFixed(0)}}%`;

      const vStrip = document.getElementById("evVerdictStrip");
      vStrip.className = res.isRisk ? "verdict-strip verdict-strip-risk" : "verdict-strip";
      document.getElementById("evVerdictSwatch").className = res.isRisk ? "swatch swatch-trap" : "swatch";
      document.getElementById("evVerdictText").textContent = res.verdict;
      document.getElementById("evConservationCheck").textContent =
        `$${{res.prod.toFixed(1)}}M - $${{res.cost.toFixed(1)}}M = ${{surSign}}$${{Math.abs(res.surplus).toFixed(1)}}M`;

      const grpLabels = {{ WR_TE: "WR/TE", DL_EDGE: "DL/EDGE", DB: "DB", OL_LB: "OL" }};
      document.getElementById("evChartTitle").textContent =
        `${{grpLabels[res.grp]}} Surplus Curve (${{ozText}})`;

      const grid = document.getElementById("historicalCompsGrid");
      grid.innerHTML = "";
      res.comps.forEach((c, idx) => {{
        const card = document.createElement("article");
        card.className = "comp-card";
        const rdStr = c.udfa ? "UDFA" : `Rd ${{c.rd}} #${{c.pk}}`;
        const f40Str = c.f40 === null ? "opt-out" : `${{c.f40.toFixed(2)}}s`;
        const sSign = c.surplus >= 0 ? "+" : "-";
        const sCls = c.surplus >= 0 ? "num-turf" : "num-crimson";
        card.innerHTML =
          `<div class="comp-head">` +
            `<span class="comp-name">${{idx + 1}}. ${{c.name}}</span>` +
            `<span class="comp-sim num">${{(c.sim * 100).toFixed(0)}}%</span>` +
          `</div>` +
          `<div class="comp-meta num">${{c.yr}} ${{c.pos}}, ${{rdStr}}, ${{f40Str}}</div>` +
          `<div class="comp-meta num">α ${{c.oz >= 0 ? "+" : ""}}${{c.oz.toFixed(2)}}σ, ${{c.snaps}} snaps</div>` +
          `<div class="comp-stats num">` +
            `<span>Realized EV</span>` +
            `<strong class="${{sCls}}">${{sSign}}$${{Math.abs(c.surplus).toFixed(1)}}M</strong>` +
          `</div>`;
        grid.appendChild(card);
      }});

      drawDraftEvChart(res);
    }}

    document.querySelectorAll('input[name="ev-pos-group"]').forEach((radio) => {{
      radio.addEventListener("change", () => {{
        if (radio.checked) {{
          evState.grp = radio.value;
          renderDraftEvAll();
        }}
      }});
    }});

    pickSlider.addEventListener("input", () => {{
      evState.pick = parseInt(pickSlider.value, 10) || 1;
      renderDraftEvAll();
    }});

    optSlider.addEventListener("input", () => {{
      evState.oz = parseFloat(optSlider.value) || 0;
      renderDraftEvAll();
    }});

    document.querySelectorAll("[data-preset-grp]").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        evState.grp = btn.dataset.presetGrp;
        evState.pick = parseInt(btn.dataset.presetPick, 10);
        evState.oz = parseFloat(btn.dataset.presetOz);
        const targetRadio = document.querySelector(`input[name="ev-pos-group"][value="${{evState.grp}}"]`);
        if (targetRadio) targetRadio.checked = true;
        pickSlider.value = String(evState.pick);
        optSlider.value = String(evState.oz);
        renderDraftEvAll();
      }});
    }});

    if (typeof ResizeObserver !== "undefined") {{
      const ro = new ResizeObserver(() => {{
        drawFieldGhost();
        drawSaliencyTimeline();
        renderDraftEvAll();
      }});
      ro.observe(fieldCanvas);
      ro.observe(evCanvas);
    }}

    updateMatchupMetaUI();
    renderSaliencyAll();
    renderDraftEvAll();
    if (!REDUCED_MOTION) {{
      stopPlayback();
    }}
  }})();
  </script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
