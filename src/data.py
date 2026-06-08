"""Load and prepare international match data for PyMC models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "results.csv"


@dataclass(frozen=True)
class ModelingArrays:
    """Arrays and mappings passed into PyMC model builders."""

    home_team_array: np.ndarray
    away_team_array: np.ndarray
    home_goals_array: np.ndarray
    away_goals_array: np.ndarray
    num_teams: int
    team_to_id: dict[str, int]
    id_to_team: dict[int, str]
    n_matches: int
    n_dropped_missing_scores: int
    df: pd.DataFrame


def load_results_csv(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it first (see notebook cell 1)."
        )
    return pd.read_csv(path)


def prepare_matches(
    data_config: dict[str, Any],
    data_path: Path | None = None,
) -> ModelingArrays:
    """Filter matches, drop null scores, and build team ID arrays."""
    df = load_results_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    n_raw = len(df)
    years = int(data_config.get("years", 50))
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df["date"] >= cutoff].copy()

    if data_config.get("exclude_friendlies", True):
        df = df[df["tournament"] != "Friendly"].copy()

    n_before_drop = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    n_dropped = n_before_drop - len(df)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = np.sort(pd.concat([df["home_team"], df["away_team"]]).unique())
    team_to_id = {team: i for i, team in enumerate(all_teams)}
    df["home_team_id"] = df["home_team"].map(team_to_id)
    df["away_team_id"] = df["away_team"].map(team_to_id)
    id_to_team = {i: team for team, i in team_to_id.items()}

    return ModelingArrays(
        home_team_array=df["home_team_id"].values,
        away_team_array=df["away_team_id"].values,
        home_goals_array=df["home_score"].values,
        away_goals_array=df["away_score"].values,
        num_teams=len(all_teams),
        team_to_id=team_to_id,
        id_to_team=id_to_team,
        n_matches=len(df),
        n_dropped_missing_scores=n_dropped,
        df=df,
    )


def load_modeling_arrays(
    config: dict[str, Any],
    data_path: Path | None = None,
) -> ModelingArrays:
    """Entry point used by the experiment runner."""
    return prepare_matches(config.get("data", {}), data_path=data_path)
