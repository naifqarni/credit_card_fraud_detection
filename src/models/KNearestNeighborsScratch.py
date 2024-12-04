import numpy as np
from collections import Counter

class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        """
        Store the training data and labels.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        """
        Predict the class for each test instance.
        """
        X_test = np.array(X_test)
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        """
        Predict the class for a single instance.
        """
        # Compute Euclidean distances
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # Find the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]