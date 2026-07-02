from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    # https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/data
    raw_data_path      : str = "data/raw/atp_tennis.csv"
    
    raw_training_data_path : str = "data/training/atp_tennis_training.csv"
    raw_testing_data_path  : str = "data/testing/atp_tennis_testing.csv"

    prepared_training_data_path : str = "data/prepared/tennis_training.csv"

    models_path        : str = "models/"
    
    elo_state_path     : str = "data/prepared/elo_state.json"

    min_date           : str = "2006-01-01"
    max_date           : str = "2026-12-31"
    
    # TODO: Compute ELO until last training date

    epsilon            : float = 0.001
    
    def get_model_path(self, model_name: str) -> str:
        return self.models_path + f"{model_name}_model.pkl"

# http://www.tennis-data.co.uk/notes.txt
class FeatureConfig(BaseModel):
    raw: list[str] = [
        "Player_1",
        "Player_2",
        "Winner",
        "Rank_1",
        "Rank_2",
        "Pts_1",
        "Pts_2",
        "Odd_1",
        "Odd_2",
        "Court",
        "Surface",
        "Date",
        "Series",
        "Round",
        "Tournament",
        # "Best of"
    ]

    numeric: list[str] = [
        "Log_Rank_Diff",
        "Log_Pts_Diff",
        "Odds_Logit_Diff",
        # "Is_Best_Of_5",
    ]
    
    elo: list[str] = [
        "Elo_Diff",
        "Surface_Elo_Diff",
    ]
    
    categorical: list[str] = [
        "Court", "Surface",
        "Series", "Round",
    ]

    high_cardinality_categorical: list[str] = [
        "Tournament",
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

"""
Learning algorithms with tuned hyperparameters:
- Random Forest
- XGBoost
"""

class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    criterion: str = "gini"
    random_state: int = 42
    bootstrap: bool = True
    max_features: str | None = "sqrt"
    n_jobs: int = 1
    
class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    objective: str = "binary:logistic" # probability output for binary classification
    eval_metric: str = "logloss"
    tree_method: str = "hist" # tree construction algorithm, faster for large datasets

    random_state: int = 42
    n_jobs: int = 1
    verbosity: int = 0

# Configuration for cross-validation and hyperparameters fine-tuning
class SearchConfig(BaseModel):
    n_iter: int = 100
    cv: int = 5
    scoring: str = "neg_log_loss"   # accuracy / neg_log_loss
    n_jobs: int = -1            # Use all available cores for parallel processing
    random_state: int = 42
    verbose: int = 1

class TrainingConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42

    search: SearchConfig = SearchConfig()
    random_forest: RandomForestConfig = RandomForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
