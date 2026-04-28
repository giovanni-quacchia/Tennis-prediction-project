from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from tennis.config import DecisionTreeConfig, FeatureConfig
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import GridSearchCV

class TennisDecisionTree:
    def __init__(self, config: DecisionTreeConfig):
        self.config = config
        self.model = DecisionTreeClassifier(
            max_leaf_nodes=config.leafs,        # max number of leafs in the tree
            random_state=config.random_state
        )

    def tune_hyperparameters(self, X, y, preprocessor):
        param_grid = {
            "model__max_leaf_nodes": [5, 10, 20, 50, None],
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_split": [2, 5, 10],                 # min samples to split an internal node
            "model__min_samples_leaf": [1, 5, 10],                  # min samples to be at a leaf node
            "model__criterion": ["gini", "entropy"],                # how measure impurity split
            "model__ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02],    # cost complexity pruning
        }

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X, y)

        self.fitted_pipeline = grid_search.best_estimator_

        return grid_search.best_params_, grid_search.best_score_
    
    def train(self, training_set):

        features = FeatureConfig()

        X_train, y_train = training_set[features.trainable], training_set["y"]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, features.numeric),
            ("cat", cat_pipeline, features.categorical)
        ], verbose_feature_names_out=False)

        self.fitted_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        return self.tune_hyperparameters(X_train, y_train, preprocessor)
    
    def predict(self, X):
        return self.fitted_pipeline.predict(X)