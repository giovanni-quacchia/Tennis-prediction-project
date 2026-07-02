from enum import Enum

class ModelName(str, Enum):
    RANDOM_FOREST       = "random-forest"
    XGBOOST             = "xgboost"
    ENSEMBLE            = "ensemble"
    
class PredictMode(str, Enum):
    dataset  = "dataset"
    players  = "players"