import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tennis_prediction.config import DataConfig, EloConfig, FeatureConfig


def read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() == ".xlsx":
        # Some source workbooks contain unsupported formatting extensions.
        # They do not affect the tabular values used by the model.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unknown extension is not supported",
                category=UserWarning,
                module="openpyxl",
            )
            return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file format: {path.suffix}. Expected .xlsx"
    )


def merge_raw_datasets(
    source: str | Path,
    destination: str | Path,
    min_date: str,
    max_date: str,
) -> pd.DataFrame:
    """Merge the required columns from annual tennis-data.co.uk files."""
    source = Path(source)
    destination = Path(destination)
    features = FeatureConfig()

    min_year = pd.Timestamp(min_date).year
    max_year = pd.Timestamp(max_date).year
    
    # consider only <year>.xlsx file    
    files = sorted([
        path
        for path in source.glob("*.xlsx")
        if path.stem.isdigit()
        and min_year <= int(path.stem) <= max_year
    ])

    if not files:
        raise ValueError(f"No annual datasets found in {source}")

    datasets: list[pd.DataFrame] = []
    for file in files:
        dataset = read_dataframe(file)
        missing = set(features.raw) - set(dataset.columns)
        if missing:
            raise ValueError(
                f"{file.name} is missing columns: {sorted(missing)}"
            )
        datasets.append(dataset[features.raw])

    merged = pd.concat(datasets, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date", "Winner", "Loser"])
    merged = merged[
        merged["Date"].between(min_date, max_date, inclusive="both")
    ]
    merged = merged.drop_duplicates(
        subset=["Date", "Tournament", "Round", "Winner", "Loser"],
        keep="last",
    )
    merged = merged.sort_values("Date", kind="stable").reset_index(drop=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(destination, index=False)
    return merged


def create_temporal_train_test_split(
    source: str | Path,
    training_path: str | Path,
    testing_path: str | Path,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest matches as an untouched temporal test set."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset = read_dataframe(source)
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    if dataset["Date"].isna().any():
        raise ValueError("Merged dataset contains invalid dates")

    dataset = dataset.sort_values("Date", kind="stable").reset_index(drop=True)
    split_index = int(len(dataset) * (1 - test_size))
    if split_index <= 0 or split_index >= len(dataset):
        raise ValueError("Temporal split produced an empty dataset")

    # Keep all matches from the split date together on the test side.
    split_date = dataset.iloc[split_index]["Date"]
    training = dataset[dataset["Date"] < split_date].copy()
    testing = dataset[dataset["Date"] >= split_date].copy()

    for path, frame in ((training_path, training), (testing_path, testing)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_excel(path, index=False)

    return training, testing


@dataclass
class HistoricalState:
    overall_ratings: dict[str, float] = field(default_factory=dict)
    surface_ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    recent_elo_ratings: dict[str, list[float]] = field(default_factory=dict)
    recent_results: dict[str, list[int]] = field(default_factory=dict)
    recent_first_set_results: dict[str, list[int]] = field(
        default_factory=dict
    )
    recent_match_loads: dict[
        str,
        list[tuple[pd.Timestamp, float]],
    ] = field(default_factory=dict)
    head_to_head_wins: dict[str, dict[str, int]] = field(default_factory=dict)
    surface_head_to_head_wins: dict[
        str,
        dict[str, dict[str, int]],
    ] = field(default_factory=dict)


def _update_elo_pair(
    ratings: dict[str, float],
    player_1: str,
    player_2: str,
    player_1_rating: float,
    player_2_rating: float,
    player_1_score: float,
    k_factor: float,
) -> None:
    """Update two opposite Elo ratings after a match result."""
    expected_score = 1 / (
        1 + 10 ** ((player_2_rating - player_1_rating) / 400)
    )
    change = k_factor * (player_1_score - expected_score)
    ratings[player_1] = player_1_rating + change
    ratings[player_2] = player_2_rating - change


def _recent_win_rate(results: list[int]) -> float:
    """Return neutral form for players without previous matches."""
    return float(np.mean(results)) if results else 0.5


def _match_load(row) -> float:
    """Approximate match workload from games played in each set."""
    load = 0.0
    for set_number in range(1, 6):
        winner_games = getattr(row, f"W{set_number}")
        loser_games = getattr(row, f"L{set_number}")
        if pd.notna(winner_games) and pd.notna(loser_games):
            # Later sets receive more weight as accumulated effort increases.
            load += set_number * (winner_games + loser_games)
    return load


def _recent_fatigue(
    history: list[tuple[pd.Timestamp, float]],
    match_date: pd.Timestamp,
    decay_days: float,
) -> float:
    """Discount previous match loads exponentially with elapsed time."""
    return float(sum(
        np.exp(-(match_date - previous_date).days / decay_days) * load
        for previous_date, load in history
    ))


def _head_to_head_difference(
    wins: dict[str, dict[str, int]],
    player_1: str,
    player_2: str,
) -> float:
    """Return Player 1's historical win-rate advantage over Player 2."""
    player_1_wins = wins.get(player_1, {}).get(player_2, 0)
    player_2_wins = wins.get(player_2, {}).get(player_1, 0)
    matches = player_1_wins + player_2_wins
    return (
        (player_1_wins - player_2_wins) / matches
        if matches
        else 0.0
    )


def add_historical_features(
    dataset: pd.DataFrame,
    config: EloConfig,
    state: HistoricalState,
) -> pd.DataFrame:
    """Calculate features using only information available before each match."""
    dataset = dataset.sort_values("Date", kind="stable").copy()
    features = FeatureConfig()
    feature_values: dict[str, list[float | int]] = {
        "Elo_Diff": [],
        "Surface_Elo_Diff": [],
        "Elo_Progression_Diff": [],
        "Recent_Form_Diff": [],
        "Recent_First_Set_Form_Diff": [],
        "Fatigue_Diff": [],
        "H2H_Diff": [],
        "H2H_Surface_Diff": [],
    }

    for row in dataset.itertuples(index=False):
        player_1, player_2 = row.Player_1, row.Player_2
        player_1_score = float(row.Winner == player_1)
        match_date = pd.Timestamp(row.Date)

        # Missing ratings start from the same neutral Elo value.
        overall_ratings = (
            state.overall_ratings.get(player_1, config.initial_rating),
            state.overall_ratings.get(player_2, config.initial_rating),
        )
        elo_histories = (
            state.recent_elo_ratings.get(player_1, []),
            state.recent_elo_ratings.get(player_2, []),
        )
        elo_progressions = tuple(
            rating - history[0] if history else 0.0
            for rating, history in zip(overall_ratings, elo_histories)
        )
        surface = str(row.Surface)
        surface_ratings = state.surface_ratings.setdefault(surface, {})
        surface_rating_values = (
            surface_ratings.get(player_1, config.initial_rating),
            surface_ratings.get(player_2, config.initial_rating),
        )

        # Store a bounded window of previous results for each player.
        recent_histories = (
            state.recent_results.get(player_1, []),
            state.recent_results.get(player_2, []),
        )
        recent_forms = tuple(map(_recent_win_rate, recent_histories))
        first_set_histories = (
            state.recent_first_set_results.get(player_1, []),
            state.recent_first_set_results.get(player_2, []),
        )
        first_set_forms = tuple(
            map(_recent_win_rate, first_set_histories)
        )

        # Retain only workloads inside the configured acute-fatigue window.
        fatigue_histories = []
        for player in (player_1, player_2):
            history = state.recent_match_loads.get(player, [])
            history = [
                (previous_date, load)
                for previous_date, load in history
                if 0 <= (match_date - previous_date).days
                <= features.fatigue_window_days
            ][-features.history_window :]
            state.recent_match_loads[player] = history
            fatigue_histories.append(history)
        fatigue_scores = tuple(
            _recent_fatigue(
                history,
                match_date,
                features.fatigue_decay_days,
            )
            for history in fatigue_histories
        )

        # Count only previous meetings between these exact opponents.
        h2h_difference = _head_to_head_difference(
            state.head_to_head_wins,
            player_1,
            player_2,
        )
        surface_h2h_wins = state.surface_head_to_head_wins.setdefault(
            surface, {}
        )
        surface_h2h_difference = _head_to_head_difference(
            surface_h2h_wins,
            player_1,
            player_2,
        )

        # Every directional feature follows Player 1 minus Player 2.
        feature_values["Elo_Diff"].append(
            overall_ratings[0] - overall_ratings[1]
        )
        feature_values["Surface_Elo_Diff"].append(
            surface_rating_values[0] - surface_rating_values[1]
        )
        feature_values["Elo_Progression_Diff"].append(
            elo_progressions[0] - elo_progressions[1]
        )
        feature_values["Recent_Form_Diff"].append(
            recent_forms[0] - recent_forms[1]
        )
        feature_values["Recent_First_Set_Form_Diff"].append(
            first_set_forms[0] - first_set_forms[1]
        )
        feature_values["Fatigue_Diff"].append(
            fatigue_scores[0] - fatigue_scores[1]
        )
        feature_values["H2H_Diff"].append(h2h_difference)
        feature_values["H2H_Surface_Diff"].append(surface_h2h_difference)

        # Keep pre-match Elo snapshots to measure momentum without leakage.
        for player, rating, history in zip(
            (player_1, player_2), overall_ratings, elo_histories
        ):
            state.recent_elo_ratings[player] = (
                history + [rating]
            )[-features.history_window :]

        # Update global and surface Elo only after recording pre-match features.
        _update_elo_pair(
            state.overall_ratings,
            player_1,
            player_2,
            *overall_ratings,
            player_1_score,
            config.k_factor,
        )
        _update_elo_pair(
            surface_ratings,
            player_1,
            player_2,
            *surface_rating_values,
            player_1_score,
            config.k_factor,
        )

        # Append the current outcome only after recording pre-match form.
        for player, result, history in (
            (player_1, int(player_1_score), recent_histories[0]),
            (player_2, int(1 - player_1_score), recent_histories[1]),
        ):
            state.recent_results[player] = (
                history + [result]
            )[-features.history_window :]

        # First-set scores update only future matches, never the current row.
        if pd.notna(row.W1) and pd.notna(row.L1) and row.W1 != row.L1:
            first_set_winner = row.Winner if row.W1 > row.L1 else row.Loser
            for player, history in (
                (player_1, first_set_histories[0]),
                (player_2, first_set_histories[1]),
            ):
                result = int(player == first_set_winner)
                state.recent_first_set_results[player] = (
                    history + [result]
                )[-features.history_window :]

        # The completed match contributes equal workload to both players'
        # future fatigue, because both played every game counted above.
        current_match_load = _match_load(row)
        if current_match_load > 0:
            for player, history in zip(
                (player_1, player_2), fatigue_histories
            ):
                state.recent_match_loads[player] = (
                    history + [(match_date, current_match_load)]
                )[-features.history_window :]

        # Record the winner only after exposing both pre-match H2H features.
        loser = player_2 if row.Winner == player_1 else player_1
        for wins in (state.head_to_head_wins, surface_h2h_wins):
            winner_h2h = wins.setdefault(row.Winner, {})
            winner_h2h[loser] = winner_h2h.get(loser, 0) + 1

    # Assign all historical columns together after the chronological scan.
    dataset = dataset.assign(**feature_values)
    return dataset


def prepare_data(
    raw_dataset: pd.DataFrame,
    historical_state: HistoricalState | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create leakage-free model features from winner/loser source rows."""
    features = FeatureConfig()
    data_config = DataConfig()
    historical_state = historical_state or HistoricalState()

    missing = set(features.raw) - set(raw_dataset.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")

    dataset = raw_dataset[features.raw].copy()
    dataset["Date"] = pd.to_datetime(dataset["Date"], errors="coerce")
    if dataset["Date"].isna().any():
        raise ValueError("Raw dataset contains invalid dates")
    dataset = dataset.sort_values("Date", kind="stable").reset_index(drop=True)

    numeric_source_columns = [
        "WRank",
        "LRank",
        "WPts",
        "LPts",
        "B365W",
        "B365L",
        "W1",
        "L1",
        "W2",
        "L2",
        "W3",
        "L3",
        "W4",
        "L4",
        "W5",
        "L5",
        "Best of",
    ]
    for column in numeric_source_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    for column in ["WRank", "LRank", "WPts", "LPts", "B365W", "B365L"]:
        dataset[column] = dataset[column].where(dataset[column] > 0)
    for column in [
        "W1", "L1", "W2", "L2", "W3", "L3",
        "W4", "L4", "W5", "L5",
    ]:
        dataset[column] = dataset[column].where(dataset[column] >= 0)

    # Random orientation prevents Winner from becoming an implicit target feature.
    rng = np.random.default_rng(random_state)
    winner_is_player_1 = rng.random(len(dataset)) < 0.5
    dataset["Player_1"] = np.where(
        winner_is_player_1, dataset["Winner"], dataset["Loser"]
    )
    dataset["Player_2"] = np.where(
        winner_is_player_1, dataset["Loser"], dataset["Winner"]
    )

    for output, winner_column, loser_column in (
        ("Rank", "WRank", "LRank"),
        ("Pts", "WPts", "LPts"),
        ("Odd", "B365W", "B365L"),
    ):
        # Keep every source value aligned with its randomized player role.
        dataset[f"{output}_1"] = np.where(
            winner_is_player_1,
            dataset[winner_column],
            dataset[loser_column],
        )
        dataset[f"{output}_2"] = np.where(
            winner_is_player_1,
            dataset[loser_column],
            dataset[winner_column],
        )

    # The target is 1 exactly when the randomized Player 1 won the match.
    dataset["y"] = winner_is_player_1.astype(int)

    # Log differences emphasize changes near the top of the rankings.
    dataset["Log_Rank_Diff"] = (
        np.log(dataset["Rank_1"]) - np.log(dataset["Rank_2"])
    )

    # log1p supports zero points while reducing the scale of large totals.
    dataset["Log_Pts_Diff"] = (
        np.log1p(dataset["Pts_1"]) - np.log1p(dataset["Pts_2"])
    )

    # Normalize inverse odds to remove the bookmaker overround.
    inverse_1 = 1 / dataset["Odd_1"]
    inverse_2 = 1 / dataset["Odd_2"]
    market_probability = inverse_1 / (inverse_1 + inverse_2)
    market_probability = market_probability.clip(
        data_config.epsilon, 1 - data_config.epsilon
    )

    # Log-odds provide an unbounded, symmetric market-strength feature.
    dataset["Odds_Logit_Diff"] = np.log(
        market_probability / (1 - market_probability)
    )

    # Best-of-five is represented as a compact binary match-format feature.
    dataset["Is_Best_Of_5"] = dataset["Best of"].eq(5).astype(int)

    # Historical features are computed last, in chronological order.
    dataset = add_historical_features(
        dataset,
        EloConfig(),
        historical_state,
    )
    return dataset[
        features.trainable + features.debug + ["y"]
    ].copy()
