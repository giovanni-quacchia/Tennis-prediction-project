from scipy.stats import randint
from tennis.config import FeatureConfig, RandomForestConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from tennis.models.base import BaseModel

class TennisRandomForest(BaseModel):
    def __init__(self, config: RandomForestConfig):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_leaf_nodes=config.max_leaf_nodes,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            criterion=config.criterion,
            random_state=config.random_state,
            bootstrap=config.bootstrap
        )
        
    def tune_hyperparameters(self, X, y, preprocessor):
        param_dist = {
            "model__n_estimators": randint(50, 500),        # scipy.stats.random is a distribution        
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__max_leaf_nodes": [None, 10, 20, 50],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__criterion": ["gini", "entropy"],
            "model__bootstrap": [True, False],
            "model__max_features": ["sqrt", "log2", None]     
        }

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model)
        ])

        random_search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=param_dist,
            n_iter=50,          # budget: max number of combinations to try
            cv=5,
            scoring="accuracy",
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X, y)
        self.fitted_pipeline = random_search.best_estimator_

        return random_search.best_params_, random_search.best_score_

    
    def train(self, training_set):
        features = FeatureConfig()

        X_train, y_train = training_set[features.trainable], training_set["y"]

        # Preprocessing pipeline for numeric and categorical features
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, features.numeric),
            ("cat", cat_pipeline, features.categorical)
        ], verbose_feature_names_out=False)

        # Tune hyperparameters using GridSearchCV
        best_params, best_score = self.tune_hyperparameters(X_train, y_train, preprocessor)

        return best_params, best_score
    
    def predict(self, X):
        return self.fitted_pipeline.predict(X)