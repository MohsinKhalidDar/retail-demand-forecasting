"""Train and evaluate demand models with a chronological holdout."""
from __future__ import annotations

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from config import METRICS_PATH, MODEL_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, DATABASE_PATH
from src.database import save_sales
from src.evaluation import metrics
from src.feature_engineering import FEATURE_COLUMNS, create_features
from src.preprocessing import make_demo_data, validate_and_clean


def _models():
    models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=250, max_depth=7, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror", random_state=42, n_jobs=1)
    except ImportError:
        models["Gradient Boosting"] = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=42)
    return models


def train(raw_path=RAW_DATA_PATH) -> dict:
    raw = validate_and_clean(pd.read_csv(raw_path) if raw_path.exists() else make_demo_data())
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    save_sales(raw, DATABASE_PATH)
    featured = create_features(raw).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if featured.empty:
        raise ValueError("Not enough history to create lag features. Provide at least 31 dated records per series.")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(PROCESSED_DATA_PATH, index=False)
    cutoff = featured.date.quantile(0.8)
    train_df, test_df = featured[featured.date <= cutoff], featured[featured.date > cutoff]
    if train_df.empty or test_df.empty:
        raise ValueError("Not enough dated observations for a chronological train/test split.")
    results, best_name, best_model, best_score = {}, "", None, float("inf")
    for name, model in _models().items():
        model.fit(train_df[FEATURE_COLUMNS], train_df.sales)
        score = metrics(test_df.sales, model.predict(test_df[FEATURE_COLUMNS]))
        results[name] = score
        if score["rmse"] < best_score:
            best_name, best_model, best_score = name, model, score["rmse"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURE_COLUMNS, "model_name": best_name}, MODEL_PATH)
    payload = {"selected_model": best_name, "holdout_start": str(cutoff.date()), "models": results}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
