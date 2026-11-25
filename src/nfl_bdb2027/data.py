"""100% Polars lazy and vectorized loaders for official NFL Big Data Bowl 2027 CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .config import DEFAULT_DATA_DIR, NA_VALUES


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved official dataset file paths on disk."""

    root: Path
    players: Path
    combine_results: Path
    combine_tracking: Path
    career_successes: Path
    games: Path
    player_play: Path
    game_tracking_files: tuple[Path, ...]

    @classmethod
    def from_dir(cls, data_dir: Path | str = DEFAULT_DATA_DIR) -> "DatasetPaths":
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Official NFL BDB 2027 data directory not found: {root}")
        gt_files = tuple(sorted(root.glob("game_tracking_*.csv")))
        if not gt_files:
            raise FileNotFoundError(f"No game_tracking_*.csv files found in {root}")
        paths = cls(
            root=root,
            players=root / "players.csv",
            combine_results=root / "combine_results.csv",
            combine_tracking=root / "combine_tracking.csv",
            career_successes=root / "player_career_successes.csv",
            games=root / "games.csv",
            player_play=root / "player_play.csv",
            game_tracking_files=gt_files,
        )
        for required in (
            paths.players,
            paths.combine_results,
            paths.combine_tracking,
            paths.career_successes,
            paths.games,
            paths.player_play,
        ):
            if not required.exists():
                raise FileNotFoundError(f"Missing required official CSV: {required}")
        return paths


def load_players(paths: DatasetPaths) -> pl.DataFrame:
    return pl.read_csv(paths.players, null_values=NA_VALUES, infer_schema_length=10000)


def load_combine_results(paths: DatasetPaths) -> pl.DataFrame:
    return pl.read_csv(paths.combine_results, null_values=NA_VALUES, infer_schema_length=10000)


def load_career_successes(paths: DatasetPaths) -> pl.DataFrame:
    return pl.read_csv(paths.career_successes, null_values=NA_VALUES, infer_schema_length=10000)


def load_games(paths: DatasetPaths) -> pl.DataFrame:
    return pl.read_csv(paths.games, null_values=NA_VALUES, infer_schema_length=10000)


def scan_combine_tracking(paths: DatasetPaths) -> pl.LazyFrame:
    return pl.scan_csv(paths.combine_tracking, null_values=NA_VALUES, infer_schema_length=10000)


def scan_player_play(paths: DatasetPaths) -> pl.LazyFrame:
    return pl.scan_csv(paths.player_play, null_values=NA_VALUES, infer_schema_length=10000)


def scan_game_tracking(paths: DatasetPaths) -> pl.LazyFrame:
    """Lazily scan all available game_tracking_*.csv files without blowing RAM."""
    return pl.scan_csv(
        [str(p) for p in paths.game_tracking_files],
        null_values=NA_VALUES,
        infer_schema_length=2000,
    )


def audit_dataset_inventory(paths: DatasetPaths) -> dict[str, Any]:
    """Compute exact verified row counts and coverage directly from official CSVs."""
    players = load_players(paths)
    cr = load_combine_results(paths)
    cs = load_career_successes(paths)
    games = load_games(paths)

    ct_lf = scan_combine_tracking(paths)
    ct_counts = (
        ct_lf.group_by("entity_type")
        .agg(pl.len().alias("rows"))
        .collect()
        .to_dicts()
    )
    ct_by_entity = {r["entity_type"]: int(r["rows"]) for r in ct_counts}
    ct_player_lf = ct_lf.filter(pl.col("entity_type") == "PLAYER")
    ct_cov = (
        ct_player_lf.group_by("nfl_id")
        .agg([
            (pl.col("drill_type") == "FORTY_YARD_DASH").any().alias("has_40"),
            (pl.col("drill_type") == "THREE_CONE_DRILL").any().alias("has_3c"),
            (pl.col("drill_type") == "SHORT_SHUTTLE").any().alias("has_ss"),
            pl.col("drill_type").str.starts_with("SKILL_DRILLS_").any().alias("has_skill"),
        ])
        .select([
            pl.len().alias("players_tracked"),
            pl.col("has_40").sum().alias("players_with_40_tracking"),
            pl.col("has_3c").sum().alias("players_with_3cone_tracking"),
            pl.col("has_ss").sum().alias("players_with_shuttle_tracking"),
            pl.col("has_skill").sum().alias("players_with_skill_tracking"),
            (pl.col("has_3c") | pl.col("has_ss")).sum().alias("players_with_any_agility_tracking"),
        ])
        .collect()
        .to_dicts()[0]
    )

    pp_stats = (
        scan_player_play(paths)
        .select([
            pl.len().alias("rows"),
            pl.col("nfl_id").n_unique().alias("unique_players"),
        ])
        .collect()
        .to_dicts()[0]
    )

    gt_per_file = {}
    gt_total_rows = 0
    for gf in paths.game_tracking_files:
        n_rows = int(pl.scan_csv(gf, null_values=NA_VALUES, infer_schema_length=1000).select(pl.len()).collect().item())
        gt_per_file[gf.name] = n_rows
        gt_total_rows += n_rows
    gt_unique_players = int(scan_game_tracking(paths).select(pl.col("nfl_id").n_unique()).collect().item())

    draft_year_counts = {
        str(r["draft_year"]): int(r["len"])
        for r in players.group_by("draft_year").len().sort("draft_year").to_dicts()
    }
    pos_counts = {
        str(r["combine_position"]): int(r["len"])
        for r in cr.group_by("combine_position").len().sort("len", descending=True).to_dicts()
    }

    return {
        "data_dir": str(paths.root),
        "total_prospects": int(players.height),
        "draft_year_counts": draft_year_counts,
        "combine_position_counts": pos_counts,
        "udfa_count": int(players["draft_overall_pick"].null_count()),
        "combine_results_rows": int(cr.height),
        "forty_opt_out_count": int(cr["forty"].null_count()),
        "three_cone_opt_out_count": int(cr["three_cone"].null_count()),
        "short_shuttle_opt_out_count": int(cr["short_shuttle"].null_count()),
        "career_successes_rows": int(cs.height),
        "games_rows": int(games.height),
        "combine_tracking_total_rows": int(sum(ct_by_entity.values())),
        "combine_tracking_player_rows": int(ct_by_entity.get("PLAYER", 0)),
        "combine_tracking_ball_rows": int(ct_by_entity.get("BALL", 0)),
        "combine_tracking_coverage": {k: int(v) for k, v in ct_cov.items()},
        "player_play_rows": int(pp_stats["rows"]),
        "player_play_unique_players": int(pp_stats["unique_players"]),
        "game_tracking_files": gt_per_file,
        "game_tracking_total_rows": int(gt_total_rows),
        "game_tracking_unique_players": gt_unique_players,
    }
