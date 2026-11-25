# Beyond the Stopwatch · NFL Big Data Bowl 2027

Reproducible, 100% real-data sports biomechanics analytics package, Colab NVIDIA L4 GPU PyTorch Multi-Scale 1D-CNN + Attention encoder, Economist / NYT pure-white publication graphics (SVG + 2x Retina PNG), interactive 10 Hz Canvas 2D Film-Room showcase, 1080p Scouting Film-Room Reel, and submission-ready Kaggle Writeup for **NFL Big Data Bowl 2027**.

## Headline Real-Data Results (Exact 4-Decimal Zero-Leakage OOF Metrics)

- **Sunday NFL Game Kinematic Power (`n = 497`, 22.65M frames):** Stratified 5-Fold `R² = +0.2398` (`ρ = +0.5093`) vs Stopwatch `+0.1560` (`ΔR² = +0.0838`, `p < 0.0005`) and v1 Tabular GBDT `+0.2180` (`ΔR² = +0.0218`, `p = 0.0045`); Temporal LOYO `R² = +0.2049` (`ρ = +0.4817`) vs Stopwatch `+0.0982` (`ΔR² = +0.1067`, `p < 0.0005`, 2.09× unseen-year lift).
- **DL/EDGE Regular-Season First-Step Get-Off (`n = 115`):** Stratified `R² = +0.5432` (`ρ = +0.7347`) vs Stopwatch `+0.4316` (`ΔR² = +0.1116`, `p = 0.0025`).
- **Exposure-Normalized NFL Career Value (`n = 510`):** Lifts NGS Pre-Draft Scores to Stratified `R² = +0.3439` (`ρ = +0.5798`, `Δρ = +0.0103`, `p = 0.0475`) and Temporal LOYO `R² = +0.3400` (`ΔR² = +0.0147`, `p = 0.0110`).

## Repository Layout

```
code/nfl-bdb2027/
  LICENSE
  .gitignore
  pyproject.toml
  README.md
  src/nfl_bdb2027/
    config.py       # Official dataset paths, canonical 2023–2025 drill aliases, pure-white palette
    data.py         # Polars lazy/vectorized CSV loaders (auto-globs game_tracking_*.csv)
    kinematics.py   # 10 Hz physics derivatives: signed a_tan, jerk, circular omega, curvature, CBI, impulse
    outcomes.py     # 22.65M NFL game tracking frames + 314k plays + exposure-normalized career value & alpha
    models.py       # Zero-leakage Stratified 5-Fold & Temporal LOYO CV + Colab 1D-CNN deep meta-fusion
    replay.py       # Real 10 Hz Combine + Sunday NFL Film-Room trajectory & attention bundle extractor
    charts.py       # 5 Economist / NYT pure-white vector SVG figures (#FFFFFF + #111111 + #184E32 + #CC2929)
    report.py       # Generators for real_data_metrics_v2.json, 4-decimal Kaggle Writeup, Interactive HTML Report
    cli.py          # Single-command reproducible pipeline runner
  scripts/
    train_temporal_cnn_colab.py  # PyTorch Multi-Scale 1D-CNN + Gated Temporal & MIL Attention trainer (NVIDIA L4)
    run_colab_gpu_job.sh         # Automated Colab GPU runner with EXIT-trap instant session termination
    export_figures_png.py        # 2x Retina PNG exporter for all 5 pure-white SVGs via Playwright Chromium
    render_filmroom_reel.py      # 1080p 30fps Scouting Film-Room Reel + synced SRT generator (-14 LUFS)
  notebooks/
    nfl_bdb2027_kaggle_notebook.ipynb
  tests/
    test_kinematics.py
    test_models.py
    test_artifacts.py
```

## Quickstart & Full Reproduction

```bash
# Run complete real-data pipeline on official CSVs (~55s on Apple Silicon)
PYTHONPATH=src /Volumes/Extra/venvs/wave2-ml/bin/python -m nfl_bdb2027.cli --n-boot 2000

# Export all 5 pure-white SVGs to 2x Retina PNGs for Kaggle Writeup
/Users/eligo/.local/bin/python3 scripts/export_figures_png.py

# Render 1080p Scouting Film-Room Reel + synced English SRT
/Users/eligo/.local/bin/python3 scripts/render_filmroom_reel.py

# Run automated unit, zero-leakage temporal CV, and artifact integrity tests
/Volumes/Extra/venvs/wave2-ml/bin/python -m pytest -v
```

Set `NFL_BDB2027_DATA_DIR` (or pass `--data-dir`) to point to `/kaggle/input/nfl-big-data-bowl-2027` or any local copy of the 9 official competition CSV files.

## Outputs (`../../materials/`)

- `real_data_metrics_v2.json` — Verified dataset inventory, out-of-fold ablation metrics, 95% bootstrap CIs, Colab GPU 1D-CNN metadata, transfer matrix, archetype signatures, and named player discoveries.
- `figures/fig1_kinematic_transfer_matrix.{svg,png}` — Combine-to-NFL Kinematic Transfer Matrix (2x Retina PNG).
- `figures/fig2_optical_velocity_decel_profiles.{svg,png}` — Real 10 Hz optical velocity & route-braking overlays.
- `figures/fig3_draft_capital_residual_alpha.{svg,png}` — Draft-Capital Residual Alpha across 510 rookies.
- `figures/fig4_cv_model_ablation.png` & `fig4_model_ablation_lift.{svg,png}` — Zero-leakage Stratified & Temporal LOYO ablation + quintile snap lift.
- `figures/fig5_positional_biomechanical_archetypes.{svg,png}` — Position-specific starter-vs-depth optical signatures.
- `NFL_BDB2027_Kaggle_Writeup.md` — Submission-ready 4-decimal precision Kaggle Big Data Bowl 2027 Writeup.
- `NFL_BDB2027_Interactive_Report.html` — Self-contained Economist / NYT pure-white interactive editorial showcase with live 10 Hz Canvas 2D Film-Room Trajectory Replay Player and 510-prospect explorer.
- `NFL_BDB2027_FilmRoom_Reel_1080p.mp4` & `.srt` — 1080p scouting film-room video reel with studio narration, documentary pulse soundbed (-14 LUFS), live 10 Hz animations, and burned-in English captions.
