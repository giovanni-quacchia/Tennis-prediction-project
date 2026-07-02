from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier

from tennis.config import (
    FeatureConfig,
    SearchConfig,
    XGBoostConfig,
)
from tennis.models.base import BaseModel
from tennis.preprocessing import build_tree_preprocessor


class TennisXGBoost(BaseModel):

    @property
    def param_distributions(self):
        return {
            "model__n_estimators": randint(150, 700),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 8),
            "model__min_child_weight": loguniform(0.5, 10.0),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": loguniform(1e-4, 10.0),
            "model__reg_lambda": loguniform(0.1, 20.0),
            "preprocessor__high_cardinality_cat__onehot__max_categories": [
                5,
                10,
                20,
                30,
            ],
        }

    def __init__(
        self,
        model_config: XGBoostConfig,
        search_config: SearchConfig,
    ):
        super().__init__(
            model=XGBClassifier(**model_config.model_dump()),
            search_config=search_config,
        )

    def train(self, training_set):
        features = FeatureConfig()

        X_train = training_set[features.trainable]
        y_train = training_set["y"]

        preprocessor = build_tree_preprocessor(features)

        return self.tune_hyperparameters(
            X_train,
            y_train,
            preprocessor,
        )
