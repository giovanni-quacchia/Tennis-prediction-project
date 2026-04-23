import math
from xml.parsers.expat import model

from sklearn.pipeline import Pipeline

from tennis.config import KNNConfig
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

from tennis.config import FeatureConfig

class TennisKNN:
    def __init__(self, config: KNNConfig):
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            p=config.p
        )

    def tune_hyperparameters(self, X, y, preprocessor):
        param_grid = {
            "model__n_neighbors": [i for i in range(1, 60, 2)],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2] # Manhattan - Euclidean distance
        }

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=10,
            scoring="accuracy",
            n_jobs=-1
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

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, features.numeric),
            ("cat", cat_pipeline, features.categorical)
        ], verbose_feature_names_out=False)

        return self.tune_hyperparameters(X, y, preprocessor)

    def predict(self, X):
        return self.model.predict(X)
    

