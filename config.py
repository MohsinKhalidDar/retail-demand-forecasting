from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "retail_sales.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "retail_features.csv"
DATABASE_PATH = BASE_DIR / "database" / "sales.db"
MODEL_PATH = BASE_DIR / "models" / "demand_forecaster.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"
RANDOM_SEED = 42
