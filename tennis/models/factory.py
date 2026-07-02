from tennis.enums import ModelName
from tennis.models.random_forest import TennisRandomForest

def build_model(model_name: ModelName, training_config):
    if model_name == ModelName.RANDOM_FOREST:
        return TennisRandomForest(
            model_config=training_config.random_forest,
            search_config=training_config.search,
        )
    
    if model_name == ModelName.XGBOOST:
        from tennis.models.xgboost import TennisXGBoost
        return TennisXGBoost(
            model_config=training_config.xgboost,
            search_config=training_config.search,
        )

    raise ValueError(f"Model {model_name.value} not implemented yet.")