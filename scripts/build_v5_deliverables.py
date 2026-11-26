"""Build all V5 deliverables:
1. Calibrate NFL GM Draft-Room Surplus EV Engine on all 510 official prospects + V4 OOF scores.
2. Refresh all 5 Economist/NYT SVGs with V4 real data and generate new Figure 6.
3. Export all 6 figures (plus fig4_cv_model_ablation.png alias) at 2x Retina PNG.
4. Generate upgraded pure-white `materials/NFL_BDB2027_Interactive_FilmRoom.html`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent

from nfl_bdb2027.charts import generate_all_figures
from nfl_bdb2027.config import ARTIFACTS_DIR, FIGURES_DIR, MATERIALS_DIR
from nfl_bdb2027.data import DatasetPaths, load_combine_results
from nfl_bdb2027.draft_ev import DraftSurplusEVEngine
from nfl_bdb2027.filmroom_v5 import write_v5_interactive_filmroom
from nfl_bdb2027.kinematics import aggregate_player_kinematics, extract_rep_level_kinematics
from nfl_bdb2027.outcomes import build_outcomes_table


def _load_v4_oof_scores() -> dict[int, np.ndarray]:
    npz_path = ARTIFACTS_DIR / "v4_positional_attn_gpu_oof.npz"
    if not npz_path.exists():
        return {}
    z = np.load(npz_path, allow_pickle=True)
    nfl_ids = z["nfl_ids"]
    sub_idx = z["v4_b1_strat_sub_indices"]
    oof_feats = z["v4_b1_strat_oof_feats"]
    scores: dict[int, np.ndarray] = {}
    for row_i, orig_idx in enumerate(sub_idx):
        vec = np.asarray(oof_feats[row_i], dtype=np.float64)
        if np.all(np.isfinite(vec)):
            scores[int(nfl_ids[int(orig_idx)])] = vec
    return scores


def main() -> None:
    t0 = time.perf_counter()
    paths = DatasetPaths.from_dir()

    v2_metrics = json.loads((MATERIALS_DIR / "real_data_metrics_v2.json").read_text(encoding="utf-8"))
    v4_metrics = json.loads((MATERIALS_DIR / "real_data_metrics_v4_gpu.json").read_text(encoding="utf-8"))
    v4_saliency = json.loads((MATERIALS_DIR / "v4_saliency_profiles.json").read_text(encoding="utf-8"))

    print("[V5] Extracting rep-level kinematics & outcomes for all 510 prospects...")
    rep_df = extract_rep_level_kinematics(paths)
    cr_pl = load_combine_results(paths)
    outcomes_df = build_outcomes_table(paths)
    full_kin = aggregate_player_kinematics(rep_df, cr_pl, fit_nfl_ids=None).to_pandas()
    master_df = outcomes_df.merge(full_kin, on="nfl_id", how="left")

    v4_oof_scores = _load_v4_oof_scores()
    print(f"[V5] Fitting DraftSurplusEVEngine (n={len(master_df)}, V4 OOF scores={len(v4_oof_scores)})...")
    ev_engine = DraftSurplusEVEngine.fit(master_df, v4_oof_scores=v4_oof_scores)
    ev_bundle = ev_engine.export_ui_and_chart_bundle()
    master_df["optical_alpha_z"] = ev_engine.prospects_df["optical_alpha_z"].to_numpy()

    # Verify sample GM scenarios
    for grp, pk, oz in [("WR_TE", 177, 1.8), ("DL_EDGE", 54, 1.4), ("DB", 24, -1.4)]:
        ev = ev_engine.evaluate(grp, pk, oz)
        print(
            f"  [EV] {grp} #{pk} (Rd {ev['draft_round']}, {oz:+.1f}σ): "
            f"Surplus {ev['expected_surplus_ev_m']:+.2f}M "
            f"(Prod ${ev['expected_production_4yr_m']:.2f}M - Cost ${ev['rookie_contract_4yr_m']:.2f}M) | "
            f"Bust Red {ev['bust_risk_reduction_pct']:+.1f}% | Top Comp: {ev['historical_comps'][0]['display_name']}"
        )

    print("[V5] Generating all 6 pure-white Economist/NYT SVG figures...")
    svgs = generate_all_figures(
        paths=paths,
        results=v2_metrics["modeling_results"],
        master_df=master_df,
        figures_dir=FIGURES_DIR,
        v4_metrics=v4_metrics,
        v4_saliency=v4_saliency,
        ev_bundle=ev_bundle,
    )
    for k, p in svgs.items():
        print(f"  [SVG] {k}: {p.name} ({p.stat().st_size:,} bytes)")

    print("[V5] Exporting all 6 SVG figures (+ fig4 alias) to 2x Retina PNGs...")
    pw_py = shutil.which("python3") or sys.executable
    subprocess.run(
        [pw_py, str(SCRIPTS_DIR / "export_figures_png.py"), "--figures-dir", str(FIGURES_DIR), "--scale", "2"],
        check=True,
    )

    print("[V5] Generating upgraded Interactive Film-Room Studio HTML...")
    html_path = write_v5_interactive_filmroom(
        paths=paths,
        v4_metrics=v4_metrics,
        v4_saliency=v4_saliency,
        ev_bundle=ev_bundle,
        out_path=MATERIALS_DIR / "NFL_BDB2027_Interactive_FilmRoom.html",
    )
    print(f"  [HTML] {html_path.name} ({html_path.stat().st_size:,} bytes)")
    print(f"[V5] Complete in {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
