"""Data loading, validation, and deterministic demo-data generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import RANDOM_SEED

REQUIRED_COLUMNS = {"date", "store_id", "product_id", "sales"}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common retail CSV schemas to this application's schema."""
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    aliases = {"date": "date", "store": "store_id", "storeid": "store_id", "product": "product_id",
               "item_id": "product_id", "sales": "sales", "revenue": "sales", "promo": "promotion",
               "customers": "customers", "stateholiday": "holiday"}
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns})
    return frame


def validate_and_clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalize_columns(frame).drop_duplicates().copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Required columns missing: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["date", "store_id", "product_id", "sales"])
    frame = frame[frame["sales"] >= 0]
    for col in ("promotion", "holiday"):
        if col not in frame:
            frame[col] = 0
        frame[col] = frame[col].fillna(0).astype(int).clip(0, 1)
    if "customers" not in frame:
        frame["customers"] = np.nan
    return frame.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def adapt_uploaded_schema(
    frame: pd.DataFrame,
    date_column: str,
    metric_column: str,
    primary_group: str | None = None,
    secondary_group: str | None = None,
) -> pd.DataFrame:
    """Map a generic time-series CSV into the canonical forecasting schema.

    A file only needs a date and a numeric metric. One or two grouping columns
    optionally create separate forecast series (for example store/product,
    region/category, or country/channel).
    """
    output = pd.DataFrame({"date": frame[date_column], "sales": frame[metric_column]})
    for source, target, fallback in ((primary_group, "store_id", "All"), (secondary_group, "product_id", "All")):
        values = frame[source].fillna("Unknown") if source else pd.Series(fallback, index=frame.index)
        # Forecasting estimators need numeric inputs; preserve a stable category code.
        output[target] = pd.factorize(values.astype(str), sort=True)[0] + 1
    for optional in ("promotion", "holiday", "customers"):
        if optional in frame.columns:
            output[optional] = frame[optional]
    cleaned = validate_and_clean(output)
    if cleaned.empty:
        raise ValueError("The selected date and metric columns contain no usable date/numeric values.")
    return cleaned


def make_demo_data(days: int = 540, stores: int = 5, products: int = 8) -> pd.DataFrame:
    """Create realistic daily product-store demand for an out-of-the-box demo."""
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    rows = []
    for store in range(1, stores + 1):
        for product in range(1, products + 1):
            base = rng.integers(80, 240)
            for i, date in enumerate(dates):
                promotion = int(rng.random() < 0.18)
                holiday = int(date.month == 12 and date.day in {24, 25, 31})
                weekly = 1.25 if date.dayofweek in (4, 5) else 0.88 if date.dayofweek == 0 else 1
                annual = 1 + 0.18 * np.sin(2 * np.pi * i / 365)
                demand = base * weekly * annual * (1.28 if promotion else 1) * (1.1 if holiday else 1)
                sales = max(0, round(demand + rng.normal(0, base * 0.09), 2))
                rows.append((date, store, product, sales, promotion, holiday, max(1, int(sales / rng.uniform(2.3, 4.5)))))
    return pd.DataFrame(rows, columns=["date", "store_id", "product_id", "sales", "promotion", "holiday", "customers"])
