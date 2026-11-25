"""Configuration, canonical drill mappings, and Economist/NYT editorial visual tokens."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = CODE_ROOT / "artifacts"
MATERIALS_DIR = PROJECT_ROOT / "materials"
FIGURES_DIR = MATERIALS_DIR / "figures"

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "NFL_BDB2027_DATA_DIR",
        "/Volumes/Extra/datasets/kaggle-nfl2027/nfl-big-data-bowl-2027",
    )
)

NA_VALUES = ["NA", ""]
SAMPLING_HZ = 10.0
DT_SEC = 1.0 / SAMPLING_HZ
LBS_TO_KG = 0.45359237
UDFA_IMPUTED_PICK = 265.0
LATEST_SEASON_PLUS_ONE = 2026  # 2023->3 seasons, 2024->2 seasons, 2025->1 season

# Empirically verified cross-year drill naming aliases in combine_tracking.csv (2023-2025).
# Canonicalizing these eliminates artificial temporal covariate shift across draft classes.
DRILL_NAME_ALIASES: dict[str, str] = {
    "LINE": "LINE_DRILL",
    "PASS_PRO_MIRROR_DRILL": "PASS_PRO-MIRROR_DRILL",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "SHORT_ZONE_BREAK_FORWARD": "SHORT_ZONE_BREAKS",
    "SHORT_ZONE_BREAK_INSIDE": "SHORT_ZONE_BREAKS",
    "SHORT_ZONE_BREAK_FLAT_WHEEL": "SHORT_ZONE_BREAKS",
}

COMBINE_POSITIONS = ["WR", "DB", "DL", "OL", "TE"]

# Pure-White Economist / NYT Data-Journalism Editorial Palette.
# Strictly pure white ground (#FFFFFF), jet ink (#111111), NFL turf forest green (#184E32),
# editorial alert crimson (#CC2929), and honor antique gold (#C8922E).
# Zero warm beige/parchment; zero blue, cyan, teal, or purple hues.
PALETTE: dict[str, str] = {
    "paper": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_alt": "#F7F7F7",
    "surface_tint": "#F2F2F2",
    "rule": "#E2E2E2",
    "rule_strong": "#111111",
    "turf": "#184E32",
    "turf_mid": "#2E6F4E",
    "turf_pale": "#EAF2EE",
    "crimson": "#CC2929",
    "crimson_dark": "#A81E1E",
    "crimson_pale": "#FCEEEE",
    "gold": "#C8922E",
    "gold_text": "#875E12",
    "gold_pale": "#FBF4E4",
    "espresso": "#3D352E",
    "olive": "#556B2F",
    "ink": "#111111",
    "ink_muted": "#4A4A4A",
    "ink_faint": "#666666",
    # Backward-compatible aliases kept pure-white/neutral so legacy keys never reintroduce beige
    "parchment": "#FFFFFF",
    "card": "#FFFFFF",
    "sand": "#EBEBEB",
    "terracotta": "#CC2929",
    "terracotta_mid": "#D94A4A",
    "terracotta_pale": "#FCEEEE",
    "brass": "#C8922E",
    "brass_mid": "#D9A84A",
    "brass_pale": "#FBF4E4",
    "sage": "#556B2F",
    "charcoal": "#111111",
}

POSITION_COLORS: dict[str, str] = {
    "WR": "#CC2929",
    "DB": "#184E32",
    "DL": "#9A6B16",
    "OL": "#3D352E",
    "TE": "#556B2F",
}

POSITION_LABELS: dict[str, str] = {
    "WR": "Wide Receivers (WR)",
    "DB": "Defensive Backs (DB)",
    "DL": "Defensive Line / Edge (DL)",
    "OL": "Offensive Line (OL)",
    "TE": "Tight Ends (TE)",
}
