from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from tennis_prediction.config import FeatureConfig, SearchConfig, XGBoostConfig
from tennis_prediction.preprocessing import build_preprocessor


class TennisXGBoost:
    def __init__(
        self,
        model_config: XGBoostConfig,
        search_config: SearchConfig,
    ):
        self.model_config = model_config
        self.search_config = search_config
        self.fitted_pipeline: Pipeline | None = None

    @property
    def parameter_distributions(self) -> dict:
        return {
            "model__n_estimators": randint(150, 700),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(2, 8),
            "model__min_child_weight": loguniform(0.5, 10.0),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": loguniform(1e-4, 10.0),
            "model__reg_lambda": loguniform(0.1, 20.0),
            "preprocessor__tournament__onehot__max_categories": [
                5,
                10,
                15,
                20,
                25,
                30,
            ],
        }

    def train(self, training_set):
        features = FeatureConfig()
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(features)),
            ("model", XGBClassifier(**self.model_config.model_dump())),
        ])
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=self.parameter_distributions,
            cv=TimeSeriesSplit(n_splits=self.search_config.cv),
            refit=True,
            error_score="raise",
            **self.search_config.model_dump(exclude={"cv"}),
        )
        search.fit(
            training_set[features.trainable],
            training_set["y"],
        )
        self.fitted_pipeline = search.best_estimator_
        return search.best_params_, search.best_score_
