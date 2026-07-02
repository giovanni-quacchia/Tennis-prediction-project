import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tennis.config import (
    DataConfig,
    EloConfig, 
    FeatureConfig,
)

# Read dataframe from CSV or Excel file
def read_dataframe(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {suffix}")

# https://www.pecan.ai/blog/data-preparation-for-machine-learning/
def create_temporal_train_test_split(
    src_path: str,
    training_path: str,
    testing_path: str,
    min_date: str,
    max_date: str,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    df = read_dataframe(src_path)
    
    # sort data to train on past matches and test on future matches
    # convert to datetime (excel may store dates as strings)
    if "Date" not in df.columns:
        raise ValueError("Dataset does not contain a Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = int(df["Date"].isna().sum())
    if invalid_dates:
        raise ValueError(
            f"Found {invalid_dates} invalid dates in the dataset"
        )
            
    # filter out matches before min_date and after max_date
    df = (
        df[df["Date"].between(min_date, max_date)]
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    
    if df.empty:
        raise ValueError(
            f"No matches found between {min_date} and {max_date}"
        )
        
    # Split training - test sets
    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Temporal split produced an empty dataset")

    split_date = df.iloc[split_index]["Date"]
    
    training_set = df[df["Date"] < split_date].copy()
    testing_set = df[df["Date"] >= split_date].copy()

    if training_set.empty or testing_set.empty:
        raise ValueError("Temporal split produced an empty dataset")
            
    Path(training_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(testing_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    training_set.to_csv(training_path, index=False)
    testing_set.to_csv(testing_path, index=False)

    return training_set, testing_set

def validate_raw_data(
    raw_df: pd.DataFrame,
    required_columns: list[str]
) -> pd.DataFrame:
    df = raw_df.copy()

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns on raw data: {missing_columns}"
        )

    # Convert Date (string in excel) to datetime
    # coerce: invalid parsing will be set as NaT (Not a Time)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    invalid_dates = df["Date"].isna().sum()
    if invalid_dates:
        raise ValueError(
            f"Found {invalid_dates} rows with invalid dates"
        )

    return df


@dataclass
class EloState:
    overall_ratings: dict[str, float] = field(
        default_factory=dict
    )
    surface_ratings: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    last_processed_date: str | None = None


def save_elo_state(
    state: EloState,
    path: str,
) -> None:
    payload = {
        "schema_version": 1,
        "last_processed_date": state.last_processed_date,
        "overall_ratings": state.overall_ratings,
        "surface_ratings": state.surface_ratings,
    }

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )
        
def load_elo_state(path: str) -> EloState:
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported feature state version"
        )

    return EloState(
        overall_ratings=payload["overall_ratings"],
        surface_ratings=payload["surface_ratings"],
        last_processed_date=payload["last_processed_date"],
    )


def add_elo_features(
    df: pd.DataFrame,
    config: EloConfig,
    state: EloState,
) -> pd.DataFrame:
    df = df.sort_values("Date", kind="stable").copy()

    elo_differences: list[float] = []
    surface_elo_differences: list[float] = []

    # iterate over rows in order of date to update Elo ratings after each match
    for row in df.itertuples(index=False):
        player_1 = row.Player_1
        player_2 = row.Player_2
        if row.Winner not in {player_1, player_2}:
            raise ValueError(
                f"Winner {row.Winner!r} is not one of the players"
            )

        player_1_score = float(row.Winner == player_1)
        player_1_elo = state.overall_ratings.get(
            player_1,
            config.initial_rating,
        )
        player_2_elo = state.overall_ratings.get(
            player_2,
            config.initial_rating,
        )

        surface = str(row.Surface)

        surface_ratings = state.surface_ratings.setdefault(
            surface,
            {},
        )

        player_1_surface_elo = surface_ratings.get(
            player_1,
            config.initial_rating,
        )
        player_2_surface_elo = surface_ratings.get(
            player_2,
            config.initial_rating,
        )

        # Compute features before updating ratings with this result.
        elo_differences.append(player_1_elo - player_2_elo)
        surface_elo_differences.append(
            player_1_surface_elo - player_2_surface_elo
        )

        player_1_expected = 1 / (
            1 + 10 ** ((player_2_elo - player_1_elo) / 400)
        )
        elo_change = config.k_factor * (
            player_1_score - player_1_expected
        )

        state.overall_ratings[player_1] = player_1_elo + elo_change
        state.overall_ratings[player_2] = player_2_elo - elo_change

        player_1_surface_expected = 1 / (
            1
            + 10
            ** (
                (player_2_surface_elo - player_1_surface_elo)
                / 400
            )
        )
        surface_change = config.k_factor * (
            player_1_score - player_1_surface_expected
        )

        surface_ratings[player_1] = (
            player_1_surface_elo + surface_change
        )
        surface_ratings[player_2] = (
            player_2_surface_elo - surface_change
        )

    # Assign new features in one assignment
    df["Elo_Diff"] = elo_differences
    df["Surface_Elo_Diff"] = surface_elo_differences

    # Update last processed date in EloState
    if not df.empty:
        state.last_processed_date = (
            df["Date"].max().date().isoformat()
        )

    return df


"""
Shape: (2644, 38)

0 duplicates
"""
def prepare_data(
    raw_df: pd.DataFrame,
    elo_state: EloState | None = None,
) -> pd.DataFrame:

    features = FeatureConfig()
    data_config = DataConfig()
    elo_state = elo_state or EloState()
        
    raw_df = validate_raw_data(
        raw_df,
        required_columns=features.raw,
    )

    df = raw_df[features.raw].copy()


    raw_numeric_cols = [
        "Rank_1",
        "Rank_2",
        "Pts_1",
        "Pts_2",
        "Odd_1",
        "Odd_2",
    ]
    for column in raw_numeric_cols:
        df[column] = pd.to_numeric(
            df[column], 
            errors="coerce" # invalid parsing will be set as NaN
        )
        
    # kaggle dataset uses -1 to indicate missing values, we replace them with NaN
    df[raw_numeric_cols] = df[raw_numeric_cols].replace(
        -1,
        np.nan,
    )

    # --- Feature engineering ---
    df["y"] = (
        df["Winner"] == df["Player_1"]
    ).astype(int)

    rank_p1, rank_p2 = df["Rank_1"], df["Rank_2"]
    # df["Rank_Diff"] = rank_p1 - rank_p2  
    df["Log_Rank_Diff"] = np.log(rank_p1) - np.log(rank_p2)  # log(rankp1/rankp2)
   
    pts_p1, pts_p2 = df["Pts_1"], df["Pts_2"]
    df["Log_Pts_Diff"] = np.log1p(pts_p1) - np.log1p(pts_p2)  # log(1+ptsp1) - log(1+ptsp2)
    # inverse odds
    player_1_inverse = 1 / df["Odd_1"]
    player_2_inverse = 1 / df["Odd_2"]
    probability_total = player_1_inverse + player_2_inverse

    market_probability = (
        player_1_inverse / probability_total
    )

    # clip: enclose values in range [epsilon, 1-epsilon] to avoid logit of 0 or 1
    market_probability = market_probability.clip(
        data_config.epsilon,
        1 - data_config.epsilon,
    )

    # logit: log(p / (1 - p))
    df["Odds_Logit_Diff"] = np.log(
        market_probability
        / (1 - market_probability)
    )
    
    elo_config = EloConfig()
    
    df = add_elo_features(
        df,
        config=elo_config,
        state=elo_state,
    )

    # --- Selection ---
    df = df[
        features.numeric 
        + features.categorical 
        + features.high_cardinality_categorical
        + features.elo
        + features.debug 
        + ["y"]
    ]
    
    return df
