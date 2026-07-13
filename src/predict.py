from __future__ import annotations

import joblib
import pandas as pd

from config import MODEL_PATH
from src.feature_engineering import create_features


def forecast(history: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """Recursive forecasts for every store-product series in the supplied history."""
    artifact = joblib.load(MODEL_PATH)
    model, columns = artifact["model"], artifact["features"]
    working = history.copy()
    working["date"] = pd.to_datetime(working["date"])
    results = []
    for _ in range(days):
        next_date = working.date.max() + pd.Timedelta(days=1)
        future = working.groupby(["store_id", "product_id"], as_index=False).tail(1).copy()
        future["date"], future["sales"], future["promotion"], future["holiday"] = next_date, 0.0, 0, 0
        candidate = pd.concat([working, future], ignore_index=True)
        engineered = create_features(candidate)
        newest = engineered[engineered.date == next_date].copy()
        newest["sales"] = model.predict(newest[columns]).clip(0)
        working = pd.concat([working, newest[history.columns]], ignore_index=True)
        results.append(newest[["date", "store_id", "product_id", "sales"]])
    return pd.concat(results, ignore_index=True)
