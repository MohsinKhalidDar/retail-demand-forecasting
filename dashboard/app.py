"""Polished interactive dashboard for tabular time-series forecasting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import METRICS_PATH, MODEL_PATH, RAW_DATA_PATH
from src.database import load_sales
from src.predict import forecast
from src.preprocessing import adapt_uploaded_schema
from src.train import train

st.set_page_config(page_title="Demand Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #172033, #111827);
        border: 1px solid #2d3c57;
        border-top: 3px solid #3b82f6;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, .18);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f8fafc !important; }
    [data-testid="stMetricDelta"] { font-weight: 600; }
    .hero { padding: 1.3rem 1.5rem; border-radius: 16px; color: white; background: linear-gradient(115deg, #0f172a, #1d4ed8); margin-bottom: 1.4rem; }
    .hero h1 { margin: 0; font-size: 2rem; } .hero p { margin: .35rem 0 0; color: #dbeafe; }
    .stDownloadButton button { width: 100%; }
    .guide-card { background: #111827; border: 1px solid #2d3c57; border-radius: 12px; padding: 1rem; min-height: 90px; }
    .guide-card b { color: #bfdbfe; } .guide-card p { margin: .35rem 0 0; color: #cbd5e1; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not MODEL_PATH.exists():
        train()
    return load_sales(ROOT / "database" / "sales.db")


def currency(value: float) -> str:
    return f"${value:,.0f}"


def upload_controls() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("1️⃣ Upload your data")
    st.sidebar.caption("Start with a CSV that has a date and a number you want to predict.")
    uploaded_file = st.sidebar.file_uploader(
        "📁 Choose a CSV file", type="csv",
        help="Examples: daily sales, website visitors, deliveries, energy use, or leads.",
    )
    if uploaded_file is None:
        st.sidebar.info("Using the included demo dataset. Upload a file whenever you are ready.")
        return
    try:
        preview = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        st.sidebar.error(f"Unable to read this CSV: {error}")
        return
    if len(preview.columns) < 2:
        st.sidebar.error("A forecast dataset needs at least a date and metric column.")
        return
    columns = list(preview.columns)
    st.sidebar.success(f"{uploaded_file.name} · {len(preview):,} rows · {len(columns)} columns")
    with st.sidebar.expander("Preview uploaded data"):
        st.dataframe(preview.head(5), use_container_width=True, hide_index=True)
    st.sidebar.markdown("**2️⃣ Tell us what to forecast**")
    date_column = st.sidebar.selectbox("📅 When did it happen? (date column)", columns, help="Select the column containing dates, such as Date, Order Date, or Timestamp.")
    metric_column = st.sidebar.selectbox("🎯 What number should we predict?", [col for col in columns if col != date_column], help="Choose a numeric column such as Sales, Visitors, Orders, Revenue, or Energy Usage.")
    no_group = "No grouping (single series)"
    groups = [no_group] + [col for col in columns if col not in {date_column, metric_column}]
    primary_group = st.sidebar.selectbox("🏷️ Split by group (optional)", groups, help="Use this for separate series such as Store, Region, Product, or Channel. Leave it as 'No grouping' for one overall forecast.")
    secondary_group = st.sidebar.selectbox("🏷️ Add another split (optional)", [no_group] + [col for col in groups[1:] if col != primary_group], help="Optional: choose a second category, such as Product after Store.")
    if st.sidebar.button("🚀 Load data & build forecast", type="primary", use_container_width=True):
        try:
            canonical = adapt_uploaded_schema(
                preview, date_column, metric_column,
                None if primary_group == no_group else primary_group,
                None if secondary_group == no_group else secondary_group,
            )
            RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            canonical.to_csv(RAW_DATA_PATH, index=False)
            with st.spinner("Checking data quality and training models…"):
                train(RAW_DATA_PATH)
            st.cache_data.clear()
            st.toast(f"Success! Loaded {len(canonical):,} valid rows and built a new forecast.", icon="✅")
            st.rerun()
        except Exception as error:  # display a useful UI error rather than a Streamlit traceback
            st.sidebar.error(f"Training could not start: {error}")


def main() -> None:
    st.markdown("""
    <div class="hero"><h1>📊 Demand Intelligence</h1>
    <p>Turn historical data into a simple forecast and practical next steps.</p></div>
    """, unsafe_allow_html=True)
    upload_controls()
    sales = load_data()
    sales["date"] = pd.to_datetime(sales["date"])
    with st.expander("👋 New here? Follow these three steps", expanded=False):
        step1, step2, step3 = st.columns(3)
        step1.markdown("<div class='guide-card'><b>1. Upload</b><p>Choose a CSV file from the sidebar. The demo data is ready if you want to explore first.</p></div>", unsafe_allow_html=True)
        step2.markdown("<div class='guide-card'><b>2. Choose</b><p>Select the date column and the number you want to predict. Groups are optional.</p></div>", unsafe_allow_html=True)
        step3.markdown("<div class='guide-card'><b>3. Understand</b><p>Review the forecast, trends, model check, and plain-language recommendations.</p></div>", unsafe_allow_html=True)
    stores, products = sorted(sales.store_id.unique()), sorted(sales.product_id.unique())
    with st.sidebar:
        st.divider(); st.subheader("3️⃣ Explore results")
        selected_stores = st.multiselect("🏷️ Group 1", stores, default=stores, help="Choose which Group 1 values to include in charts and the forecast.")
        selected_products = st.multiselect("🏷️ Group 2", products, default=products, help="Choose which Group 2 values to include in charts and the forecast.")
        horizon = st.slider("🔮 How far ahead?", min_value=7, max_value=90, value=30, step=1, format="%d days", help="Choose how many future days the app should estimate.")
        if st.button("🔄 Rebuild forecast", use_container_width=True, help="Train the models again using the currently saved dataset."):
            try:
                with st.spinner("Retraining models…"):
                    train()
                st.cache_data.clear(); st.rerun()
            except Exception as error:
                st.error(f"Retraining failed: {error}")
    filtered = sales[sales.store_id.isin(selected_stores) & sales.product_id.isin(selected_products)].copy()
    if filtered.empty:
        st.warning("Select at least one value in each grouping filter to continue.")
        return
    daily = filtered.groupby("date", as_index=False).sales.sum()
    last_day = filtered.date.max()
    recent = filtered[filtered.date > last_day - pd.Timedelta(days=30)]
    prior = filtered[(filtered.date <= last_day - pd.Timedelta(days=30)) & (filtered.date > last_day - pd.Timedelta(days=60))]
    growth = ((recent.sales.sum() / max(prior.sales.sum(), 1)) - 1) * 100
    group_one = filtered.groupby("store_id", as_index=False).sales.sum().sort_values("sales", ascending=False)
    group_two = filtered.groupby("product_id", as_index=False).sales.sum().sort_values("sales", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total value", currency(filtered.sales.sum()), help="The sum of the selected metric across all currently selected data.")
    c2.metric("🧾 Records analysed", f"{len(filtered):,}", help="The number of rows used in the current analysis.")
    c3.metric("📅 Last 30 days", currency(recent.sales.sum()), f"{growth:+.1f}% vs prior 30d", help="Compares the most recent 30 days with the previous 30 days.")
    c4.metric("🔗 Active series", f"{filtered.groupby(['store_id', 'product_id']).ngroups:,}", help="The number of unique Group 1 + Group 2 combinations included.")
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "📈 Trends", "🧪 Model check", "💡 Insights"])
    with tab1:
        st.subheader("Your forecast")
        st.caption("Blue shows what happened in the past. Orange dashed lines show the estimated future values.")
        try:
            forecast_data = forecast(filtered, horizon)
        except Exception as error:
            st.error(f"The forecast could not be calculated: {error}")
            st.stop()
        future = forecast_data.groupby("date", as_index=False).sales.sum()
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=daily.date, y=daily.sales, name="Historical", line=dict(color="#2563eb", width=2.5)))
        figure.add_trace(go.Scatter(x=future.date, y=future.sales, name="Forecast", line=dict(color="#f97316", width=2.5, dash="dash")))
        figure.add_vline(x=last_day, line_dash="dot", line_color="#64748b")
        figure.update_layout(height=430, margin=dict(l=5, r=5, t=20, b=5), hovermode="x unified", xaxis_title=None, yaxis_title="Metric")
        st.plotly_chart(figure, use_container_width=True)
        left, right = st.columns([1, 2])
        left.metric(f"🔮 Estimated next {horizon} days", currency(future.sales.sum()))
        right.download_button("⬇️ Download forecast as CSV", forecast_data.to_csv(index=False).encode("utf-8"), "demand_forecast.csv", "text/csv", use_container_width=True)
        with st.expander("📋 View the detailed forecast"):
            st.dataframe(forecast_data.sort_values("sales", ascending=False), use_container_width=True, hide_index=True)
    with tab2:
        left, right = st.columns(2)
        monthly = daily.set_index("date").resample("ME").sum().reset_index()
        left.plotly_chart(px.line(daily, x="date", y="sales", title="Daily trend", color_discrete_sequence=["#2563eb"]), use_container_width=True)
        right.plotly_chart(px.bar(monthly, x="date", y="sales", title="Monthly performance", color_discrete_sequence=["#0f766e"]), use_container_width=True)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(group_one, x="store_id", y="sales", title="Performance by Group 1", color_discrete_sequence=["#7c3aed"]), use_container_width=True)
        b.plotly_chart(px.bar(group_two, x="product_id", y="sales", title="Performance by Group 2", color_discrete_sequence=["#db2777"]), use_container_width=True)
    with tab3:
        report = json.loads(METRICS_PATH.read_text())
        st.subheader("How trustworthy is the forecast?")
        st.caption(f"Best model: {report['selected_model']} · It was tested on later dates starting {report['holdout_start']}.")
        scores = pd.DataFrame(report["models"]).T.reset_index(names="Model")
        st.dataframe(scores, use_container_width=True, hide_index=True)
        st.info("✅ This is a fair test: models learn from earlier dates and are checked against later dates. Lower MAE/RMSE/MAPE is better; R² closer to 1 is better.")
    with tab4:
        st.subheader("What should you do next?")
        promo_sales = filtered.groupby("promotion").sales.mean()
        uplift = (promo_sales.get(1, 0) / max(promo_sales.get(0, 1), 1) - 1) * 100
        r1, r2, r3 = st.columns(3)
        r1.success(f"**Focus capacity**\n\nGroup 1 value {group_one.iloc[0].store_id} contributes the most in this selection.")
        r2.info(f"**Investigate opportunity**\n\nGroup 2 value {group_two.iloc[-1].product_id} is the lowest-performing selected segment.")
        r3.warning(f"**Promotion signal**\n\nPromotion periods average {uplift:+.0f}% versus non-promotion periods.")
        st.divider()
        st.subheader("🩺 Data health")
        d1, d2, d3 = st.columns(3)
        d1.metric("Date range", f"{filtered.date.min():%d %b %Y} – {last_day:%d %b %Y}")
        d2.metric("Days of history", f"{filtered.date.nunique():,}")
        d3.metric("Missing metric values", f"{filtered.sales.isna().sum():,}")


if __name__ == "__main__":
    main()
