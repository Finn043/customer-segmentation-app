# Customer Segmentation with K-Means and PCA

A compact, ML-focused project in one Jupyter notebook. It explores the Mall Customers dataset, standardizes features, implements K-Means and PCA from scratch with NumPy, evaluates cluster counts, and interprets the resulting customer groups.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Customer_Segmentation.ipynb
```

The notebook downloads the public dataset from Kaggle into `data/Mall_Customers.csv` the first time it runs. You can also download it manually from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python). The CSV is excluded from Git; the notebook contains the data-loading step.

## Notebook outline

1. Load and inspect the Mall Customers dataset.
2. Standardize Age, Annual Income and Spending Score. K-Means uses Euclidean distance, so unscaled features with larger ranges could dominate.
3. Implement k-means++ initialization and Lloyd's assignment/update steps in NumPy. The objective is within-cluster sum of squared errors:

   $$J = \sum_{i=1}^{m} \|x_i - \mu_{c_i}\|^2$$

4. Compare SSE over K=1…10 (Elbow method) and silhouette scores. K=5 is shown as an example, not a claim that it is the uniquely correct segmentation.
5. Implement PCA from the covariance eigendecomposition and visualize clusters in the first two principal components. K-Means still clusters in the full three-feature space.
6. Inspect cluster profiles and predict the segment for a new customer.

## Key concepts

- **Unsupervised learning:** there are no target labels; cluster IDs are discovered group assignments and need human interpretation.
- **Feature scaling:** `StandardScaler` applies `z = (x - mean) / standard_deviation`; fit on the dataset and reuse the same scaler for new points.
- **K-Means:** assignment and centroid-update steps reduce SSE but converge to a local optimum. Initialization can affect the result.
- **Elbow and silhouette:** complementary heuristics for choosing K; validate clusters with domain context as well.
- **PCA:** finds directions of maximum variance for a lower-dimensional view. PCA visualizes clusters here; it does not create them.

`CustomerID` is excluded from model features. The dataset is for learning and exploratory analysis; these clusters are not validated business segments.
