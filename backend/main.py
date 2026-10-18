from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.evaluation import elbow_scores, silhouette
from src.kmeans_scratch import KMeansScratch
from src.pca_scratch import PCAScratch
from src.preprocessing import FEATURES, load_customers, transform

DATA = Path(__file__).parent / "data" / "Mall_Customers.csv"
customers, scaler = load_customers(DATA)
X = transform(customers[FEATURES], scaler)
pca = PCAScratch(2).fit(X)
projection = pca.transform(X)

app = FastAPI(title="Customer Segmentation API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class CustomerInput(BaseModel):
    age: int = Field(ge=1, le=120)
    annual_income: float = Field(ge=0, le=1_000_000)
    spending_score: int = Field(ge=1, le=100)


def get_model(k: int):
    if not 2 <= k <= 10:
        raise HTTPException(status_code=422, detail="k must be between 2 and 10")
    return KMeansScratch(k).fit(X)


@app.get("/api/health")
def health():
    return {"status": "ok", "customers": len(customers)}


@app.get("/api/dashboard")
def dashboard(k: int = 5):
    model = get_model(k)
    labels = model.labels_
    rows = []
    for i, (_, customer) in enumerate(customers.iterrows()):
        rows.append({"id": int(customer["CustomerID"]), "pc1": float(projection[i, 0]), "pc2": float(projection[i, 1]),
                     "cluster": int(labels[i]), "age": int(customer["Age"]),
                     "income": float(customer["Annual Income (k$)" ]), "spending": int(customer["Spending Score (1-100)" ])})
    summaries = []
    for cluster in range(k):
        group = customers.iloc[np.flatnonzero(labels == cluster)]
        summaries.append({"cluster": cluster, "size": len(group), "income": round(float(group["Annual Income (k$)"].mean()), 1),
                          "spending": round(float(group["Spending Score (1-100)"].mean()), 1), "age": round(float(group["Age"].mean()), 1)})
    return {"points": rows, "clusters": summaries, "elbow": elbow_scores(X), "silhouette": silhouette(X, labels),
            "variance_explained": float(pca.explained_variance_ratio_.sum()), "k": k}


@app.post("/api/predict")
def predict(customer: CustomerInput, k: int = 5):
    model = get_model(k)
    values = np.array([[customer.age, customer.annual_income, customer.spending_score]], dtype=float)
    label = int(model.predict(transform(values, scaler))[0])
    point = pca.transform(transform(values, scaler))[0]
    return {"cluster": label, "pc1": float(point[0]), "pc2": float(point[1])}
