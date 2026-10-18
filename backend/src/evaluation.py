import numpy as np
from sklearn.metrics import silhouette_score

from .kmeans_scratch import KMeansScratch


def elbow_scores(X, k_values=range(1, 11)):
    return [{"k": k, "sse": KMeansScratch(k).fit(X).inertia_} for k in k_values]


def silhouette(X, labels):
    labels = np.asarray(labels)
    return float(silhouette_score(X, labels)) if 1 < len(np.unique(labels)) < len(X) else None
