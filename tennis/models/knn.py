from tennis.config import KNNConfig
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class TennisKNN:
    def __init__(self, config: KNNConfig):
        self.model = KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            p=config.p
        )
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
