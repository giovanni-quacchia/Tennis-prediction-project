from abc import ABC, abstractmethod # Abstract Base Class

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV


class BaseModel(ABC):
    
    
    @property
    @abstractmethod
    def param_distributions(self):
        pass
    
    
    def __init__(self, model, search_config):
        self.model = model
        self.search_config = search_config
        self.fitted_pipeline = None
    
    
    def tune_hyperparameters(self, X, y, preprocessor):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.model),
        ])
        
        # model_dump: get a ditionary representation: {k1: v1, k2: v2, ...}
        search_params = self.search_config.model_dump(
            exclude={"cv"}
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=self.param_distributions,
            # TimeSeriesSplit for cross-validation to respect data temporal order
            cv=TimeSeriesSplit(n_splits=self.search_config.cv),     
            refit=True,             # refit the best model on the whole dataset after search
            **search_params         # **: unpack dictionary as keyword arguments:  k1=v1, k2=v2, ...
        )

        search.fit(X, y)
        
        self.fitted_pipeline = search.best_estimator_
        
        return search.best_params_, search.best_score_

    @abstractmethod
    def train(self, train_set) -> tuple[dict, float]:
        pass

    def predict(self, X):
        
        if self.fitted_pipeline is None:
            raise RuntimeError("Model has not been trained yet.")
        
        return self.fitted_pipeline.predict(X)
    
    def score(self, X, y):
        if self.fitted_pipeline is None:
            raise RuntimeError("Model has not been trained yet.")
        
        return self.fitted_pipeline.score(X, y)