import numpy as np

from .kmeans_scratch import KMeansScratch
from .pca_scratch import PCAScratch


def main():
    X = np.array([[-5.0, -4], [-4, -5], [-5, -5], [5, 4], [4, 5], [5, 5]])
    model = KMeansScratch(2, random_state=2).fit(X)
    assert len(np.unique(model.labels_)) == 2
    assert model.inertia_ < 4
    assert np.array_equal(model.predict(X), model.labels_)
    projection = PCAScratch(2).fit(X).transform(X)
    assert projection.shape == (len(X), 2) and np.isfinite(projection).all()
    print("K-Means and PCA self-check passed")


if __name__ == "__main__":
    main()
