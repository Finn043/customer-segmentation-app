from pathlib import Path

from src.evaluation import elbow_scores, silhouette
from src.kmeans_scratch import KMeansScratch
from src.pca_scratch import PCAScratch
from src.preprocessing import FEATURES, load_customers, transform

DATA = Path(__file__).parent / "data" / "Mall_Customers.csv"


def main():
    data, scaler = load_customers(DATA)
    X = transform(data[FEATURES], scaler)
    print("Elbow scores (K=1..10):")
    for row in elbow_scores(X):
        print(f"K={row['k']:2}: SSE={row['sse']:.2f}")
    model = KMeansScratch(5).fit(X)
    print(f"\nSelected K=5 (classic mall customer segmentation); silhouette={silhouette(X, model.labels_):.3f}")
    pca = PCAScratch(2).fit(X)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.1%}")


if __name__ == "__main__":
    main()
