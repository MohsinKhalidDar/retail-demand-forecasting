from __future__ import annotations

import sqlite3
import pandas as pd


def save_sales(frame: pd.DataFrame, database_path) -> None:
    with sqlite3.connect(database_path) as conn:
        frame.to_sql("sales", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")


def load_sales(database_path) -> pd.DataFrame:
    with sqlite3.connect(database_path) as conn:
        return pd.read_sql_query("SELECT * FROM sales", conn, parse_dates=["date"])
