import json
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from tennis.config import EloConfig, FeatureConfig

# Read dataframe from CSV or Excel file
def read_dataframe(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file format: {suffix}")

# https://www.pecan.ai/blog/data-preparation-for-machine-learning/
def load_temporal_train_test_split(path: str, min_date: str, max_date: str, test_size: float = 0.3):
    
    df = pd.read_excel(path)
    
    # sort data to train on past matches and test on future matches
    df["Date"] = pd.to_datetime(df["Date"]) #  convert to datetime (excel may store dates as strings)
    df = df.sort_values("Date")
    
    min_date = pd.Timestamp(min_date)
    max_date = pd.Timestamp(max_date)
    df = df[df["Date"].between(min_date, max_date)] # filter out matches before min_date and after max_date

    train_set, test_set = train_test_split(
        df,
        test_size=test_size,
        shuffle=False, # temporal split, no shuffling
    )

    return train_set, test_set

# Adapt kaggle dataset to format of first used raw dataset
def adapt_kaggle_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    player_1_won = df["Winner"] == df["Player_1"]

    df["Loser"] = np.where(
        player_1_won,
        df["Player_2"],
        df["Player_1"],
    )

    df["WRank"] = np.where(
        player_1_won,
        df["Rank_1"],
        df["Rank_2"],
    )
    df["LRank"] = np.where(
        player_1_won,
        df["Rank_2"],
        df["Rank_1"],
    )

    df["WPts"] = np.where(
        player_1_won,
        df["Pts_1"],
        df["Pts_2"],
    )
    df["LPts"] = np.where(
        player_1_won,
        df["Pts_2"],
        df["Pts_1"],
    )

    df["WinnerOdds"] = np.where(
        player_1_won,
        df["Odd_1"],
        df["Odd_2"],
    )
    df["LoserOdds"] = np.where(
        player_1_won,
        df["Odd_2"],
        df["Odd_1"],
    )

    return df

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
        winner = row.Winner
        loser = row.Loser
        surface = row.Surface

        winner_elo = state.overall_ratings.get(
            winner, config.initial_rating
        )
        loser_elo = state.overall_ratings.get(
            loser, config.initial_rating
        )

        surface = str(row.Surface)

        surface_ratings = state.surface_ratings.setdefault(
            surface,
            {},
        )

        winner_surface_elo = surface_ratings.get(
            winner,
            config.initial_rating,
        )
        loser_surface_elo = surface_ratings.get(
            loser,
            config.initial_rating,
        )

        # we must add elo differences before the match is played
        elo_differences.append(winner_elo - loser_elo)
        surface_elo_differences.append(
            winner_surface_elo - loser_surface_elo
        )

        winner_expected = 1 / (
            1 + 10 ** ((loser_elo - winner_elo) / 400)
        )
        elo_change = config.k_factor * (1 - winner_expected)

        state.overall_ratings[winner] = winner_elo + elo_change
        state.overall_ratings[loser] = loser_elo - elo_change

        winner_surface_expected = 1 / (
            1
            + 10
            ** (
                (loser_surface_elo - winner_surface_elo)
                / 400
            )
        )
        surface_change = (
            config.k_factor * (1 - winner_surface_expected)
        )

        surface_ratings[winner] = (
            winner_surface_elo + surface_change
        )
        surface_ratings[loser] = (
            loser_surface_elo - surface_change
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
    elo_state = elo_state or EloState()
    
    raw_df = adapt_kaggle_data(raw_df)
    
    raw_df = validate_raw_data(
        raw_df,
        required_columns=features.raw
    )
    
    df = raw_df[features.raw].copy()

    raw_numeric_cols = ["WRank", "LRank", "WPts", "LPts", "WinnerOdds", "LoserOdds"]
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
    df["y"] = 1 # before randomization, Player 1 is always the winner

    df["Rank_Diff"] = df["WRank"] - df["LRank"]
    df["Pts_Diff"] = df["WPts"] - df["LPts"]
    
    # kaggle odds
    
    # convert odds to raw probabilities
    winner_inverse = 1 / df["WinnerOdds"]
    loser_inverse = 1 / df["LoserOdds"]
    probability_total = winner_inverse + loser_inverse

    # sum is greater than 1 because of the bookmaker's margin
    winner_probability = winner_inverse / probability_total
    loser_probability = loser_inverse / probability_total

    df["Odds_Prob_Diff"] = (
        winner_probability - loser_probability
    )
    
    # first dataset odds
    
    # df["B365_Bet_Diff"] = df["B365W"] - df["B365L"]
    # df["PS_Bet_Diff"] = df["PSW"] - df["PSL"]
    # df["Max_Bet_Diff"] = df["MaxW"] - df["MaxL"]
    # df["Avg_Bet_Diff"] = df["AvgW"] - df["AvgL"]
    
    df = add_elo_features(
        df,
        config=EloConfig(),
        state=elo_state,
    )
    
    random_swap(df)

    # --- Selection ---
    df = df[features.numeric + features.categorical + features.debug + ["y"]]
    
    return df

# --- Random Swap Player 1 and Player 2 ---
# Otherwise, the model will learn to alway pick Player 1 as the winner
def random_swap(df):
    rng = np.random.default_rng(seed=42)
    swap_mask = rng.integers(0, 2, size=len(df)).astype(bool) # Randomly swap 50% of the rows

    # Swap players randomly 50% of the time 
    df.loc[swap_mask, ["Winner", "Loser"]] = df.loc[swap_mask, ["Loser", "Winner"]].values

    diff_cols = [
        "Rank_Diff",
        "Pts_Diff",
        "Odds_Prob_Diff",
        "Elo_Diff",
        "Surface_Elo_Diff",
    ]

    for col in diff_cols:
        df.loc[swap_mask, col] = df.loc[swap_mask, col] * -1 # Invert difference for swapped rows

    df.loc[swap_mask, "y"] = 0 # P2 wins
    
