from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    # https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/data
    raw_data_path      : str | None = "data/raw/tennis.xlsx"
    prepared_data_path : str | None = "data/prepared/tennis_prepared.xlsx"
    testing_data_path  : str | None = "data/testing/tennis_testing.xlsx"

    models_path        : str | None = "models/"

    # start-end date for training data
    min_date           : str = "2004-01-01"
    train_cutoff_date  : str = "2025-04-14"

    epsilon            : float = 0.001
    
    def get_model_path(self, model_name: str) -> str:
        return self.models_path + f"{model_name}_model.pkl"

# http://www.tennis-data.co.uk/notes.txt
class FeatureConfig(BaseModel):
    raw: list[str] = [
        "WRank", "LRank",       # Ranking
        "WPts", "LPts",         # Entry points 
        "B365W", "B365L",       # Bets
        "PSW", "PSL",
        "MaxW", "MaxL",
        "AvgW", "AvgL",
        "Court", "Surface",     # Categories with natural order
        "Winner", "Loser",      # Players
        "Date"
    ]

    numeric: list[str] = [
        "Rank_Diff", "Pts_Diff",
        "B365_Bet_Diff", "PS_Bet_Diff", "Max_Bet_Diff", "Avg_Bet_Diff"
    ]

    categorical: list[str] = [
        "Court", "Surface"
    ]

    trainable: list[str] = numeric + categorical

    debug: list[str] = [
        "Winner", "Loser",       # Players after random swap
        "Date"
    ]

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

# Configuration for cross-validation and hyperparameters fine-tuning
class SearchConfig(BaseModel):
    n_iter: int = 50
    cv: int = 5
    scoring: str = "accuracy"
    n_jobs: int = -1            # Use all available cores for parallel processing
    random_state: int = 42
    verbose: int = 1

class TrainingConfig(BaseModel):
    test_size: float = 0.3
    random_state: int = 42

    search: SearchConfig = SearchConfig()
    random_forest: RandomForestConfig = RandomForestConfig()

