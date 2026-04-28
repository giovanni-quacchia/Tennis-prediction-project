from sklearn.pipeline import Pipeline

from tennis.config import KNNConfig, TrainingConfig
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from tennis.config import FeatureConfig

from sklearn.ensemble import BaggingClassifier

import numpy as np

class TennisKNN:
    def __init__(self, config: KNNConfig):
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            p=config.p
        )

    def tune_hyperparameters(self, X, y, preprocessor):
        config = TrainingConfig()
        param_grid = {
            # KNN hyperparams
            "model__estimator__n_neighbors": [1, 7, 11, 15, 25, 31, 35, 45, 51],
            "model__estimator__weights": ["uniform", "distance"],
            "model__estimator__p": [1, 2], # Manhattan - Euclidean distance

            # Ensemble hyperparams
            "model__n_estimators": [10,50], # number of submodels
            "model__max_samples": [0.5, 1.0], # 50% or 100% of the training data
        }

        # Wrap KNN model in a Bagging ensemble
        bagging_knn = BaggingClassifier(
            estimator=self.model, 
            random_state=config.random_state,
            n_jobs=-1)

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", bagging_knn)
        ])

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            verbose=1
        )

        grid_search.fit(X, y)

        self.fitted_pipeline = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_

    def train(self, training_set):

        features = FeatureConfig()

        X = training_set[features.trainable]
        y = training_set["y"]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # TODO: add ordinal encoding for tournament level
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # ignore unknown categories in test set and keep output as dense matrix
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, features.numeric),
            ("cat", cat_pipeline, features.categorical)
        ], verbose_feature_names_out=False)

        return self.tune_hyperparameters(X, y, preprocessor)

    def predict(self, X):
        return self.model.predict(X)
    

