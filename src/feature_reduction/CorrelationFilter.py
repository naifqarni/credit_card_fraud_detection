import numpy as np

class CorrelationFilter:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.keep_mask = None
        
    def fit_transform(self, X):
        # Calculate correlation matrix
        corr_matrix = np.abs(np.corrcoef(X.T))
        
        # Get upper triangle indices (excluding diagonal)
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # Find features with correlation greater than threshold
        high_corr_pairs = np.where(np.abs(corr_matrix[upper_tri]) > self.threshold)
        
        # Convert to original matrix indices and store correlated pairs
        correlated_pairs = []
        upper_tri_coords = np.triu_indices(corr_matrix.shape[0], k=1)
        for idx in range(len(high_corr_pairs[0])):
            i = upper_tri_coords[0][high_corr_pairs[0][idx]]
            j = upper_tri_coords[1][high_corr_pairs[0][idx]]
            correlation = corr_matrix[i,j]
            correlated_pairs.append((i, j, correlation))
        
        # Print highly correlated feature pairs
        print("\nHighly correlated feature pairs (correlation > {}):".format(self.threshold))
        for i, j, corr in correlated_pairs:
            print(f"Features {i} and {j}: correlation = {corr:.3f}")
        
        # For each group of correlated features, keep only one feature
        features_to_remove = set()
        processed_features = set()
        
        for i, j, _ in correlated_pairs:
            if i not in processed_features and j not in processed_features:
                # Keep feature i, remove feature j
                features_to_remove.add(j)
                processed_features.add(i)
                processed_features.add(j)
            elif i not in processed_features:
                features_to_remove.add(j)
                processed_features.add(i)
                processed_features.add(j)
            elif j not in processed_features:
                features_to_remove.add(i)
                processed_features.add(i)
                processed_features.add(j)
        
        # Create mask for features to keep
        self.keep_mask = np.ones(X.shape[1], dtype=bool)
        self.keep_mask[list(features_to_remove)] = False
        
        # Return filtered data
        return X[:, self.keep_mask]
        
    def transform(self, X):
        return X[:, self.keep_mask]