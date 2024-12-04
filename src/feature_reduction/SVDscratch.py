import numpy as np
class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.singular_values = None 
        self.V = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Calculate covariance matrix
        cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
        
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate singular values
        self.singular_values = np.sqrt(eigenvalues[:self.n_components])
        
        # Calculate U matrix
        self.U = np.dot(X_centered, eigenvectors) / np.sqrt(eigenvalues)
        self.U = self.U[:, :self.n_components]
        
        # Keep only k components of V
        self.V = eigenvectors[:, :self.n_components]
        
        return self
        
    def transform(self, X):
        # Center the data using mean from fit
        X_centered = X - self.mean
        
        # Transform data
        X_reduced = np.dot(X_centered, self.V)
        
        return X_reduced
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)