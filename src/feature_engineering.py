"""Leakage-safe demand forecasting features."""
from __future__ import annotations

import pandas as pd

GROUPS = ["store_id", "product_id"]
FEATURE_COLUMNS = ["store_id", "product_id", "year", "month", "quarter", "day_of_week", "weekend",
                   "promotion", "holiday", "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy().sort_values(GROUPS + ["date"])
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["quarter"] = df.date.dt.quarter
    df["day_of_week"] = df.date.dt.dayofweek
    df["weekend"] = (df.day_of_week >= 5).astype(int)
    grouped = df.groupby(GROUPS, group_keys=False)["sales"]
    for lag in (1, 7, 30):
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 14, 30):
        df[f"rolling_mean_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    return df
