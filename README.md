# Customer Segmentation & Clustering Engine

A portfolio-ready unsupervised-learning dashboard for the Mall Customers dataset. It implements Lloyd's K-Means with k-means++ initialization and PCA via NumPy, evaluates candidate cluster counts, and serves interactive insights from FastAPI to a React + TypeScript + Tailwind dashboard.

## What it does

- Scales `Age`, `Annual Income (k$)` and `Spending Score (1-100)` with `StandardScaler`.
- Runs a vectorized NumPy K-Means implementation (initialization, assignment and centroid updates).
- Projects standardized data into two principal components with NumPy SVD.
- Computes within-cluster SSE for K=1…10 and Silhouette Score for the selected K.
- Exposes cluster profiles and validated new-customer predictions through FastAPI.
- Displays the cluster map, elbow curve, segment summaries, and prediction form in a responsive dashboard.

## Project structure

```text
backend/
  data/Mall_Customers.csv  # download from Kaggle (not committed)
  src/{preprocessing,kmeans_scratch,pca_scratch,evaluation}.py
  main.py                 # FastAPI application
  train.py                # Elbow report and model metrics
frontend/
  src/components/         # Scatter/elbow charts and prediction form
  src/types/              # API payload types
  src/App.tsx             # Dashboard
netlify.toml
```

## Local development

### Dataset

The Kaggle dataset is not included in this repository. From the project root, download the public dataset into the expected path:

```bash
curl -L 'https://www.kaggle.com/api/v1/datasets/download/vjchoudhary7/customer-segmentation-tutorial-in-python' -o /tmp/mall-customers.zip
unzip -o /tmp/mall-customers.zip -d backend/data
```

You can also download it with the Kaggle CLI. Keep `Mall_Customers.csv` in `backend/data/`.

### Backend

Python 3.10+ is recommended.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                 # SSE for K=1…10 and selected-model metrics
python -m src.selfcheck         # assert-based algorithm sanity check
uvicorn main:app --reload
```

API docs: <http://localhost:8000/docs>

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Vite serves <http://localhost:5173>. For a remote API, set `VITE_API_URL` to its origin (for example, `https://your-api.example.com`) in `frontend/.env.local` before starting/building. Leave it unset for same-origin hosting.

### API

- `GET /api/health` — service status and dataset row count.
- `GET /api/dashboard?k=5` — PCA coordinates, cluster profiles, elbow SSE values and silhouette score. Valid K: 2–10.
- `POST /api/predict?k=5` — assign one customer to a cluster and return PCA coordinates.

Example request: `{"age":32,"annual_income":65,"spending_score":60}`.

## The learning behind the model

### Unsupervised learning

Supervised learning has training examples with target labels. This dataset has no ground-truth customer segments. K-Means discovers groups from feature similarity alone; cluster IDs are arbitrary labels, not known customer types. The segment profile table helps a human interpret those groups.

### Feature scaling

K-Means uses Euclidean distance. A feature with a much larger numeric range can dominate the distance and centroid assignment. `StandardScaler` applies

$$z = (x - \mu) / \sigma$$

so each feature is centered at zero and scaled to unit variance. Fit the scaler on the training data, then reuse that fitted scaler for new-customer predictions. We use the three numeric attributes; `CustomerID` is an identifier and is excluded.

### K-Means objective and optimization

For observations \(x_i\), assignments \(c_i\), and centroid \(\mu_k\), the cost (SSE / inertia) is

$$J(c,\mu) = \sum_{i=1}^{m} \|x_i - \mu_{c_i}\|^2.$$

Lloyd's algorithm alternates two steps until assignments or centers stabilize:

1. **Assignment:** assign each example to its nearest centroid, \(c_i = \arg\min_k \|x_i-\mu_k\|^2\).
2. **Centroid update:** set each centroid to the mean of its assigned examples, \(\mu_k = \frac{1}{|C_k|}\sum_{i\in C_k}x_i\).

Each step does not increase SSE, so the objective converges to a local optimum. It is not guaranteed to find a global optimum. K-means++ initialization and a fixed random seed improve starting positions and reproducibility. Empty clusters retain their previous centroid.

### Choosing K: Elbow and silhouette

The Elbow method plots SSE against K. SSE always decreases as K rises, so choose a useful tradeoff near the bend where adding clusters yields diminishing improvement. This is a heuristic, not proof of the correct K. The dashboard defaults to K=5 (a common demonstration setting for this dataset) and lets you compare K=2…10.

Silhouette score compares point `i`'s average within-cluster distance `a` with its nearest other-cluster distance `b`:

$$s = (b-a) / \max(a,b), \quad -1 \le s \le 1.$$

Scores nearer 1 indicate compact, separated groups; around 0 indicates overlap; negative values may indicate a questionable assignment. Use this with the elbow plot and domain context.

### PCA and the 2D map

PCA centers the scaled feature matrix, computes its singular value decomposition \(X=U\Sigma V^T\), and projects rows onto the first two right-singular vectors (principal directions). The scatter plot uses PC1 and PC2 for display; clustering still uses all three standardized original features. Explained variance tells how much total variation those two directions retain, so the map is a lossy view when that number is low.

## Deployment

`netlify.toml` uses `frontend` as the build base, runs `npm run build`, publishes `dist`, and routes client-side paths to `index.html`. Netlify hosts the static frontend; deploy FastAPI separately to a Python ASGI host and set the Netlify environment variable `VITE_API_URL` to its public origin, then trigger a new build. The backend currently allows cross-origin requests from any origin because this is a public demo API; restrict `allow_origins` before using private data or deploying a customer-facing system.

## Interview Q&A

**Why is K-Means unsupervised?** There are no target labels. It groups examples by feature-space distance; a person interprets the resulting groups.

**Why scale before K-Means?** Euclidean distance is sensitive to scale. Standardization prevents a wide-range feature from overwhelming the others.

**What does the K-Means cost measure?** The sum of squared distances from each observation to its assigned centroid. It measures compactness, not usefulness or correctness.

**Does K-Means find the global minimum?** No. Alternating assignment and update reaches a local optimum; initialization can change the result. K-means++ and multiple starts can reduce, but not eliminate, this issue.

**How do you select K?** Inspect diminishing SSE returns in an elbow plot, compare silhouette scores, and check whether segments are interpretable and useful. No one metric determines K in every problem.

**What is PCA optimizing?** It finds orthogonal directions of maximum variance. The first two give a 2D linear projection that retains as much variance as possible among 2D linear projections.

**Does PCA create clusters?** No. PCA is dimensionality reduction. Here it visualizes points; K-Means runs in the full standardized feature space.

**Why is explained variance relevant?** It shows how much variation is visible in the 2D map. Low retained variance means the map can hide structure present in higher dimensions.

**What are K-Means limitations?** You must choose K; it is sensitive to initialization, outliers and feature scaling; its objective favors compact roughly spherical clusters. Consider outlier handling or density/hierarchical methods if those assumptions do not fit.

## Dataset

`backend/data/Mall_Customers.csv` comes from the Kaggle Mall Customers dataset (`vjchoudhary7/customer-segmentation-tutorial-in-python`) and is intentionally excluded from this repository. `CustomerID` is never used as a model feature.
