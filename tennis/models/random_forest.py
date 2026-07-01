from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from tennis.config import FeatureConfig, RandomForestConfig, SearchConfig

from tennis.models.base import BaseModel
from tennis.preprocessing import build_tree_preprocessor

# StandardScaler non serve per random forest, ne per XGBoost, perché sono modelli basati su alberi e non su distanza, come knn

class TennisRandomForest(BaseModel):
    
    @property
    def param_distributions(self):
        return {
            "model__n_estimators": randint(50, 500),
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__max_leaf_nodes": [None, 10, 20, 50],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 10),
            "model__criterion": ["gini", "entropy"],
            "model__bootstrap": [True, False],
            "model__max_features": ["sqrt", "log2", None],
        }
    
    def __init__(
        self, 
        model_config: RandomForestConfig,
        search_config: SearchConfig
    ):
        # call superclass constructor
        super().__init__(

            # unpack dictionary as keyword arguments
            model=RandomForestClassifier(**model_config.model_dump()),
            search_config=search_config
        )
    
    def train(self, training_set):
        features = FeatureConfig()

        X_train, y_train = training_set[features.trainable], training_set["y"]

        # Preprocessing pipeline for numeric and categorical features
        preprocessor = build_tree_preprocessor(features)

        # Tune hyperparameters using RandomizedSearchCV
        best_params, best_score = self.tune_hyperparameters(X_train, y_train, preprocessor)

        return best_params, best_score