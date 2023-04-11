
import numpy as np

class SVD:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None

    def fit(self,X):
        U, D, V = np.linalg.svd(X)
        self.components = V[:self.n_components,:]
        

    def transform(self,X):
        try:
            X_tranformed = np.dot(X, self.components.T)
            return X_tranformed
        except:
            return X@self.components
    
    def fit_transform(self,X):
        U, D, V = np.linalg.svd(X)
        self.components = V[:self.n_components,:]

        try:
            X_tranformed = X.dot(self.components.T)
            return X_tranformed
        except: 
            X@self.components

    def matrix_components(self,X):
        U, D, V = np.linalg.svd(X)
        return U[:, :self.n_components], D[:self.n_components], V[:self.n_components, :]
