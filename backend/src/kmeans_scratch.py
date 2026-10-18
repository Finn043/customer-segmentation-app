import numpy as np


class KMeansScratch:
    """Vectorized Lloyd K-Means with seeded k-means++ initialization."""

    def __init__(self, n_clusters=5, max_iter=300, tol=1e-4, random_state=42):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        self.n_clusters, self.max_iter, self.tol = n_clusters, max_iter, tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < self.n_clusters or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array with at least n_clusters rows")
        rng = np.random.default_rng(self.random_state)
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            probs = distances / distances.sum() if distances.sum() else np.full(len(X), 1 / len(X))
            centers.append(X[rng.choice(len(X), p=probs)])
        self.cluster_centers_ = np.array(centers, dtype=float)
        for _ in range(self.max_iter):
            labels = self.predict(X)
            updated = self.cluster_centers_.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    updated[k] = members.mean(axis=0)
            shift = np.linalg.norm(updated - self.cluster_centers_)
            self.cluster_centers_ = updated
            if shift <= self.tol:
                break
        self.labels_ = self.predict(X)
        self.inertia_ = float(((X - self.cluster_centers_[self.labels_]) ** 2).sum())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
