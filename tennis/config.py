from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    # https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/data
    raw_data_path: str | None = "dataset/csv/tennis.xlsx"
    prepared_data_path: str | None = "dataset/csv/tennis_prepared.xlsx"
    testing_data_path: str | None = "dataset/csv/tennis_testing.xlsx"

    knn_model_path: str | None = "tennis/models/knn_model.pkl"
    decision_tree_model_path: str | None = "tennis/models/decision_tree_model.pkl"

    # start-end date for training data
    min_date: str = "2004-01-01"
    train_cutoff_date: str = "2025-04-14"

    epsilon: float = 0.001

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
- K-Nearest Neighbors
- Random Forest
"""

class KNNConfig(BaseModel):
    n_neighbors: int = 55
    weights: str = "uniform"
    p: int = 1 # euclidean distance

    test_size: float = 0.3
    random_state: int = 42

class DecisionTreeConfig(BaseModel):
    leafs: int = 10
    random_state: int = 42

class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42 
    bootstrap: bool = True

class TrainingConfig(BaseModel):
    test_size: float = 0.3
    random_state: int = 42

    knn: KNNConfig = KNNConfig()
    decision_tree: DecisionTreeConfig = DecisionTreeConfig()

    random_forest: RandomForestConfig = RandomForestConfig()

