from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_dataset_dir: str = "data/raw"
    raw_data_path: str = "data/raw/tennis_matches_combined.xlsx"
    raw_training_data_path: str = "data/training/tennis_training.xlsx"
    raw_testing_data_path: str = "data/testing/tennis_testing.xlsx"
    prepared_training_data_path: str = "data/prepared/tennis_training.xlsx"
    prepared_testing_data_path: str = "data/prepared/tennis_testing.xlsx"
    model_path: str = "models/xgboost_model.pkl"

    min_date: str = "2005-07-04"
    max_date: str = "2026-12-31"
    epsilon: float = 0.001
    random_state: int = 42


class FeatureConfig(BaseModel):
    history_window: int = 5
    fatigue_window_days: int = 10
    fatigue_decay_days: float = 3.0

    raw: list[str] = [
        # "Tournament",
        "Date",
        "Series",
        "Court",
        "Surface",
        "Round",
        "Best of",
        "Winner",
        "Loser",
        "WRank",
        "LRank",
        "WPts",
        "LPts",
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
        "B365W","B365L"
    ]

    numeric: list[str] = [
        "Log_Rank_Diff",
        "Log_Pts_Diff",
        "Odds_Logit_Diff",
        "Is_Best_Of_5",
        "Recent_Form_Diff",
        "Recent_First_Set_Form_Diff",
        "Fatigue_Diff",
        "H2H_Diff",
        "H2H_Surface_Diff",
    ]

    elo: list[str] = [
        "Elo_Diff",
        "Surface_Elo_Diff",
        "Elo_Progression_Diff",
    ]

    categorical: list[str] = [
        "Court",
        "Surface",
        "Series",
        "Round",
    ]

    high_cardinality_categorical: list[str] = [
        # "Tournament",
    ]

    trainable: list[str] = (
        numeric
        + categorical
        + high_cardinality_categorical
        + elo
    )

    debug: list[str] = [
        "Player_1",
        "Player_2",
        "Winner",
        "Date",
    ]


class EloConfig(BaseModel):
    initial_rating: float = 1500.0
    k_factor: float = 32.0


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    random_state: int = 42
    n_jobs: int = 1
    verbosity: int = 0


class SearchConfig(BaseModel):
    n_iter: int = 50
    cv: int = 5
    scoring: str = "neg_log_loss"
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 1


class TrainingConfig(BaseModel):
    test_size: float = 0.2
    search: SearchConfig = SearchConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
