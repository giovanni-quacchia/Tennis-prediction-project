from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    data_path: str | None = "dataset/csv/tennis.csv"
    output_path: str | None = "dataset/trainedData.csv"

    # start-end date for training data
    min_date: str = "2004-01-01"
    train_cutoff_date: str = "2025-04-14"

    epsilon: float = 0.001

class FeatureConfig(BaseModel):
    selected: list[str] = [
        "winner_rank",
        "loser_rank",
        "winner_elo",
        "loser_elo",
        "winner_ace",
        "loser_ace",
        "winner_df",
        "loser_df",
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

