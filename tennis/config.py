from pydantic import BaseModel, ConfigDict

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    data_path: str | None = "dataset/csv/tennis.csv"
    output_path: str | None = "dataset/traindata.csv"

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

# Three learning algorithms

class RandomForestConfig(BaseModel):
    # hyper-parameters for Random Forest
    n_estimators: int = 100
    max_depth: int | None = None
    random_state: int = 42 
    bootstrap: bool = True

class TrainingConfig(BaseModel):
    test_size: float = 0.3
    random_state: int = 42

    random_forest: RandomForestConfig = RandomForestConfig()

