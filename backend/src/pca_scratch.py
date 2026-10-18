import numpy as np


class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        self.components_ = components[: self.n_components]
        variance = singular**2 / max(len(X) - 1, 1)
        self.explained_variance_ratio_ = variance[: self.n_components] / variance.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T
