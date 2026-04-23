from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    # https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/data
    raw_data_path: str | None = "dataset/csv/tennis.xlsx"
    trained_path: str | None = "dataset/trainedData.csv"

    # start-end date for training data
    min_date: str = "2004-01-01"
    train_cutoff_date: str = "2025-04-14"

    epsilon: float = 0.001

# http://www.tennis-data.co.uk/notes.txt
class FeatureConfig(BaseModel):
    raw: list[str] = [
        "WRank", "LRank",       # Ranking
        "WPts", "LPts",         # Entry points 
        "B365W", "B365L",       # Bet365 odds
        "Court", "Surface",     # Categories with natural order
        "Winner", "Loser"       # Players
    ]

    numeric: list[str] = [
        "Rank_Diff", "Pts_Diff", "Bet_Diff"
    ]

    categorical: list[str] = [
        "Court", "Surface"
    ]

    debug: list[str] = [
        "P1", "P2",             # Players after random swap
    ]

"""
Learning algorithms with hyperparameters to be tuned:
- K-Nearest Neighbors
- Random Forest
"""

class KNNConfig(BaseModel):
    n_neighbors: int = 5
    weights: str = "distance"
    p: int = 2 # euclidean distance

    test_size: float = 0.3
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
    random_forest: RandomForestConfig = RandomForestConfig()

