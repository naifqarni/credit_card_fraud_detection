import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the logistic regression model."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights
        self.bias = 0                       # Initialize bias

        # Gradient descent
        for _ in range(self.epochs):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid to get predictions
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """Make predictions."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        # Convert probabilities to binary output (0 or 1)
        y_predicted_binary = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_binary)

    def predict_proba(self, X):
        """Return probability estimates."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)