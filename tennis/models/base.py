from abc import ABC, abstractmethod # Abstract Base Class

class BaseModel(ABC):
    @abstractmethod
    def train(self, train_set) -> tuple[dict, float]:
        pass

    @abstractmethod
    def predict(self, X):
        pass