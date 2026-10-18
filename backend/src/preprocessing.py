from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_customers(path: str | Path) -> tuple[pd.DataFrame, StandardScaler]:
    data = pd.read_csv(path)
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data.dropna(subset=FEATURES).copy()
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    scaler = StandardScaler()
    scaler.fit(data[FEATURES])
    return data, scaler


def transform(values, scaler: StandardScaler):
    if not isinstance(values, pd.DataFrame):
        values = pd.DataFrame(values, columns=FEATURES)
    return scaler.transform(values)
