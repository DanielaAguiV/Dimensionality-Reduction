
import numpy as np

class SVD:

    def __init__(self, n_components= 2):
        self.n_components = n_components
        self.components = None

    def fit(self,X):
        U, D, V = np.linalg.svd(X)
        self.components = V[:self.n_components,:]
        self.explained_variance_ = (D ** 2) / (X.shape[0] - 1)
        self.singular_values_ = D    

    def transform(self,X):
        X_tranformed = np.dot(X, self.components.T)
        return X_tranformed

    
    def fit_transform(self,X):
        U, D, V = np.linalg.svd(X)
        self.components = V[:self.n_components,:]
        X_tranformed = X.dot(self.components.T)
        return X_tranformed

    def matrix_components(self,X):
        U, D, V = np.linalg.svd(X)
        return U[:, :self.n_components], D[:self.n_components], V[:self.n_components, :]
