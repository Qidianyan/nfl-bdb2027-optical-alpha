"""End-to-end reproducible CLI runner for NFL Big Data Bowl 2027 (V2 Champion)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .charts import generate_all_figures
from .config import ARTIFACTS_DIR, DEFAULT_DATA_DIR, MATERIALS_DIR
from .data import DatasetPaths, audit_dataset_inventory, load_combine_results
from .kinematics import extract_rep_level_kinematics
from .models import run_full_modeling_suite
from .outcomes import build_outcomes_table
from .replay import build_film_room_replay_bundle
from .report import (
    write_interactive_html_report,
    write_kaggle_writeup_md,
    write_metrics_json,
)


def run_pipeline(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    materials_dir: Path | str = MATERIALS_DIR,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Execute the complete real-data NFL Big Data Bowl 2027 V2 pipeline."""
    t0 = time.time()
    paths = DatasetPaths.from_dir(data_dir)
    mat_dir = Path(materials_dir)
    art_dir = Path(artifacts_dir)
    fig_dir = mat_dir / "figures"

    print(f"[1/6] Auditing official Kaggle CSV inventory in {paths.root} ...")
    inventory = audit_dataset_inventory(paths)
    print(
        f"      510 prospects | {inventory['combine_tracking_player_rows']:,} combine player frames | "
        f"{inventory['player_play_rows']:,} plays | {inventory['game_tracking_total_rows']:,} NFL game tracking frames"
    )

    print("[2/6] Extracting 10 Hz optical Combine kinematics (Polars vectorized) ...")
    rep_df = extract_rep_level_kinematics(paths)
    cr_pl = load_combine_results(paths)
    print(f"      Extracted {rep_df.height:,} drill reps across {rep_df['nfl_id'].n_unique()} prospects")

    print("[3/6] Aggregating NFL regular-season game tracking, play outcomes & career value ...")
    outcomes_df = build_outcomes_table(paths)

    print(f"[4/6] Running zero-leakage Stratified & Temporal LOYO CV + Colab 1D-CNN fusion ({n_boot} bootstrap resamples) ...")
    results, master_df = run_full_modeling_suite(
        outcomes_df, rep_df, cr_pl, n_boot=n_boot, artifacts_dir=art_dir
    )

    print("[5/6] Extracting real 10 Hz Film-Room trajectory replay payloads (Combine + NFL Sunday plays) ...")
    replay_scenarios = build_film_room_replay_bundle(
        paths, master_df, cnn_oof_path=art_dir / "colab_temporal_cnn_oof.npz"
    )

    print("[6/6] Generating 5 pure-white NYT/Economist SVGs, metrics v2 JSON, Writeup & HTML report ...")
    figure_paths = generate_all_figures(paths, results, master_df, figures_dir=fig_dir)
    metrics_v2_path = write_metrics_json(inventory, results, out_path=mat_dir / "real_data_metrics_v2.json")
    metrics_v1_compat = write_metrics_json(inventory, results, out_path=mat_dir / "real_data_metrics.json")
    writeup_path = write_kaggle_writeup_md(
        inventory, results, master_df, out_path=mat_dir / "NFL_BDB2027_Kaggle_Writeup.md"
    )
    html_path = write_interactive_html_report(
        inventory,
        results,
        master_df,
        figure_paths,
        replay_scenarios=replay_scenarios,
        out_path=mat_dir / "NFL_BDB2027_Interactive_Report.html",
    )

    elapsed = time.time() - t0
    print(f"[DONE] Complete V2 pipeline finished in {elapsed:.1f}s.")
    return {
        "elapsed_sec": round(elapsed, 2),
        "inventory": inventory,
        "results": results,
        "paths": {
            "metrics_v2_json": str(metrics_v2_path),
            "metrics_json": str(metrics_v1_compat),
            "writeup_md": str(writeup_path),
            "interactive_html": str(html_path),
            "figures": {k: str(v) for k, v in figure_paths.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL Big Data Bowl 2027 V2 End-to-End Analytics Pipeline")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--materials-dir", type=Path, default=MATERIALS_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        materials_dir=args.materials_dir,
        artifacts_dir=args.artifacts_dir,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
