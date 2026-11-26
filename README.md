# Beyond the Stopwatch (V6 Championship Edition) · NFL Big Data Bowl 2027

Reproducible, 100% real-data sports biomechanics analytics package, **328 Colab NVIDIA L4 GPU Position-Specific Multi-Scale 1D-CNN + Dual-Head Temporal Attention encoders**, **NFL GM Draft-Room 4-Year Rookie Surplus EV Engine**, **6 Economist / NYT pure-white publication graphics (SVG + 2x Retina PNG)**, **Interactive 10 Hz Saliency Film-Room & GM EV Studio**, **V6 1080p Scouting Film-Room Reel**, and submission-ready Kaggle Writeup for **NFL Big Data Bowl 2027 ($100,000)**.

## Headline Real-Data Results (Exact 4-Decimal Zero-Leakage OOF & BH-FDR Metrics)

- **Overall V4–V6 Positional Multi-Scale Attention Fusion (`n = 497`, 22.65M NFL frames, 431,094 Combine 10 Hz frames):**
  - Stratified 5-Fold **`R² = +0.2562`** (`95% CI [+0.1954, +0.3150]`, `Spearman ρ = +0.5104`) vs V2 Unified `+0.2398` and Stopwatch `+0.1560`.
  - Unseen-year **Temporal LOYO `R² = +0.2286`** (`95% CI [+0.1659, +0.2901]`, `ρ = +0.4899`) vs V2 `+0.2049` (`ΔR² = +0.0237`, `q_FDR = 0.0264`) and Stopwatch `+0.0982` (**2.33×** unseen-year generalization, `q_FDR = 0.0012`).
  - **Incremental Partial R²:** **`+0.1188`** over Stopwatch (`q_FDR = 0.0012`) and **`+0.1121`** beyond **Draft Capital + Stopwatch** (`[+0.0653, +0.1576]`, **`q_FDR = 0.0012`**).
- **Position-Specific Encoder Breakout — `DL_EDGE` (`n = 115`):** Stratified **`R² = +0.6892`** (`[+0.6123, +0.7479]`, `ρ = +0.7816`), **Temporal LOYO `R² = +0.7045`**, **`Partial R² = +0.3566`** vs Stopwatch (`q_FDR = 0.0012`). Defensive Backs (`DB`, `n = 118`) gain `ΔR² = +0.1352` (`Partial R² = +0.1236`, `q_FDR = 0.0075`).
- **DL/EDGE First-Step Get-Off (`n = 115`) & Career Value (`n = 510`):** Deep optical fusion reaches **`R² = +0.5432`** (`ρ = +0.7347`, `p = 0.0025`) on DL Get-Off (V4 positional Partial R² `+0.1584`, `q_FDR = 0.0060`) and lifts NGS Pre-Draft Career Value to Stratified **`R² = +0.3439`** (`ρ = +0.5798`, `p = 0.0475`) / Temporal LOYO **`+0.3400`** (`p = 0.0110`).
- **NFL GM Draft-Room Surplus EV (`draft_ev.py`, `n = 510`):** Calibrated on official CBA 4-year rookie contract slots (`#1 $39.49M` → `#257 $3.96M`), elite optical prospects (`+1.5σ`) generate **`+$6.2M`** higher Day-3 4-year cap surplus and **`-20.4%`** lower bust risk vs slot-only scouting (realized surplus: Puka Nacua `+$49.0M`, Tuli Tuipulotu `+$45.4M`, Mason McCormick `+$29.8M` vs Cam Smith `-$5.3M`, Lukas Van Ness `-$5.9M`).

## Repository Layout

```text
code/nfl-bdb2027/
  LICENSE
  .gitignore
  pyproject.toml
  README.md
  src/nfl_bdb2027/
    config.py           # Official dataset paths, canonical 2023–2025 drill aliases, pure-white palette
    data.py             # Polars lazy/vectorized CSV loaders (auto-globs game_tracking_*.csv)
    kinematics.py       # 10 Hz physics derivatives: signed a_tan, jerk, circular omega, curvature, CBI, impulse
    outcomes.py         # 22.65M NFL game tracking frames + 314k plays + exposure-normalized career value & alpha
    models.py           # Zero-leakage Stratified 5-Fold & Temporal LOYO CV + V2 Colab 1D-CNN fusion
    positional_attn.py  # V4 4-position multi-scale Conv1d (k=3,5,9) + Dual-Head Temporal Attn + BH-FDR rigor
    draft_ev.py         # V5 NFL GM Draft-Room 4-Year Rookie Surplus EV Engine & 510-prospect comp retrieval
    replay.py           # Real 10 Hz Combine + Sunday NFL trajectory & attention extractor
    charts.py           # 6 Economist / NYT pure-white vector SVG figures (fig1–fig6)
    filmroom_v5.py      # V5/V6 pure-white Interactive 10 Hz Saliency Film-Room & GM Draft EV Studio generator
    report.py           # Generators for metrics JSON, 4-decimal V6 Kaggle Writeup, and V2 Interactive Report
    cli.py              # End-to-end reproducible pipeline runner
  scripts/
    train_temporal_cnn_colab.py  # V2 unified PyTorch 1D-CNN + Attention trainer (Colab L4)
    train_v4_colab_gpu.py        # V4 4-position 328-model multi-scale Dual-Head Attention trainer (Colab L4)
    run_v4_colab_gpu.sh          # Automated Colab L4 runner with EXIT-trap instant session termination
    build_v5_deliverables.py     # Builds all 6 figures (SVG + 2x PNG), V5/V6 Interactive FilmRoom HTML & V6 Writeup
    export_figures_png.py        # 2x Retina PNG exporter for all 6 pure-white SVGs via Playwright Chromium
    verify_v5_filmroom_e2e.py    # Real Chromium E2E click-through verification & screenshot capture
    render_filmroom_reel.py      # V6 1080p 30fps Scouting Film-Room Reel + synced SRT generator (-14 LUFS)
  notebooks/
    nfl_bdb2027_kaggle_notebook.ipynb
  tests/
    test_kinematics.py
    test_models.py
    test_artifacts.py
    test_positional_attn.py
    test_draft_ev.py
    test_v6_consistency.py
```

## Quickstart & Full Reproduction

```bash
# 1. Build all 6 V4–V6 figures (SVG + 2x PNG), Interactive Film-Room Studio & V6 Kaggle Writeup (~10s)
/Volumes/Extra/venvs/wave2-ml/bin/python scripts/build_v5_deliverables.py

# 2. Verify Interactive Film-Room Studio in real headless Chromium (macOS keychain-safe)
python3 scripts/verify_v5_filmroom_e2e.py

# 3. Render V6 1080p Scouting Film-Room Reel + synced English SRT (-14 LUFS)
python3 scripts/render_filmroom_reel.py

# 4. Run full automated test suite (30+ tests: zero-leakage CV, BH-FDR, EV parity, V6 4-decimal audit)
/Volumes/Extra/venvs/wave2-ml/bin/python -m pytest -v
```

Set `NFL_BDB2027_DATA_DIR` (or pass `--data-dir`) to point to `/kaggle/input/nfl-big-data-bowl-2027` or any local copy of the 9 official competition CSV files.

## Outputs (`../../materials/`)

- `real_data_metrics_v4_gpu.json` & `v4_saliency_profiles.json` — V4–V6 authoritative 4-decimal positional metrics, BH-FDR `q_FDR`, and 0.1s dual-head attention profiles.
- `real_data_metrics_v2.json` — Complete 510-prospect dataset inventory, V2 ablation suite, transfer matrix, and archetype signatures.
- `figures/fig1..fig6*.{svg,png}` — All 6 Economist / NYT pure-white 2x Retina figures (including `fig4_cv_model_ablation.png` and `fig6_temporal_attention_saliency_and_draft_ev.png`).
- `NFL_BDB2027_Kaggle_Writeup.md` — Submission-ready V6 4-decimal Kaggle Writeup embedding all 6 figures.
- `NFL_BDB2027_Interactive_FilmRoom.html` — V5/V6 pure-white Interactive 10 Hz Saliency Film-Room & GM Draft-Room Surplus EV Studio (`sloplint --render` 0 FAIL / 0 WARN).
- `NFL_BDB2027_FilmRoom_Reel_1080p.mp4` & `.srt` — V6 1080p scouting reel with studio ZONOS3-Mini (`Wyatt`) narration, documentary pulse bed (`-14 LUFS`), live 10 Hz dual-head saliency & GM EV recordings, all 6 figures, and synced English captions.
