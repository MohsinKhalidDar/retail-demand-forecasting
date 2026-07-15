# AI-Powered Retail Demand Forecasting & Inventory Analytics Platform

An end-to-end retail demand forecasting system that turns daily product-store sales into inventory-planning forecasts and business insights. It is designed as a portfolio project: reproducible pipeline, SQL persistence, chronological model evaluation, interactive dashboard, Docker image, and CI.

## Highlights

- Generates deterministic retail demo data when no CSV is supplied, so it runs immediately.
- Ingests CSV data with `date`, `store_id`, `product_id`, and `sales`; `promotion`, `holiday`, and `customers` are optional.
- Creates calendar, lag (1/7/30 day), and rolling-average (7/14/30 day) features without target leakage.
- Compares Linear Regression, Random Forest, and XGBoost (with a scikit-learn fallback), selecting the lowest chronological holdout RMSE.
- Stores cleaned sales in SQLite and surfaces demand, forecasts, model quality, and recommendations in Streamlit.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
streamlit run dashboard/app.py
```

Open `demandiq.streamlit.app`. The first training run creates a demo CSV in `data/raw/`, a SQLite database, engineered features, a model artifact, and evaluation metrics.

## Use your own data

Place `retail_sales.csv` in `data/raw/`, then run `python -m src.train`. Required columns are `date`, `store_id`, `product_id`, and `sales`. The app normalizes several common names (`Store`, `Product`, `Promo`) automatically.

You can also upload a CSV directly from the Streamlit sidebar using **Upload time-series CSV**. Select any date column, numeric metric to forecast, and optionally one or two grouping columns. This supports retail, marketing, logistics, energy, finance, and other time-series datasets; the uploaded file is mapped into the forecasting schema, updates SQLite, retrains the selected model, and refreshes the dashboard.

To test the uploader, use `sample_data/website_traffic_demo.csv` and select `date` as the date column and `website_visitors` as the metric. Leave both grouping fields set to **No grouping (single series)**.

For Rossmann data, use `Store` as `store_id`, use a stable placeholder such as `1` for `product_id` (Rossmann is store-level), and map `Sales` to `sales`, `Promo` to `promotion`, and `StateHoliday` to `holiday`.

## Quality and deployment

```bash
pytest -q
docker build -t retail-demand-forecast .
docker run -p 8501:8501 retail-demand-forecast
```

GitHub Actions runs the pipeline tests on every push and pull request.
