from sklearn.pipeline import Pipeline

from tennis.config import KNNConfig
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class TennisKNN:
    def __init__(self, config: KNNConfig):
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            p=config.p
        )

    def tune_hyperparameters(self, X, y, preprocessor):
        param_grid = {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25],
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
        return grid_search.best_params_, grid_search.best_score_

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    

