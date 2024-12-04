import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
        return self
        
    def transform(self, X):
        # Center the data using mean from fit
        X_centered = X - self.mean
        
        # Project data onto principal components
        X_transformed = np.dot(X_centered, self.components)
        
        return X_transformed
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Determine optimal number of components
def find_optimal_components(X, variance_threshold=0.95):
    # Fit PCA with all components
    pca_full = PCA(n_components=X.shape[1])
    pca_full.fit(X)
    
    # Calculate explained variance ratio
    X_transformed = pca_full.transform(X)
    explained_variance = np.var(X_transformed, axis=0)
    total_variance = np.sum(np.var(X - pca_full.mean, axis=0))
    explained_variance_ratio = explained_variance / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Find number of components needed to explain variance_threshold of variance
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    return n_components



