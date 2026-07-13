from src.feature_engineering import FEATURE_COLUMNS, create_features
from src.preprocessing import adapt_uploaded_schema, make_demo_data, validate_and_clean


def test_demo_data_is_valid_and_features_are_leakage_safe():
    raw = validate_and_clean(make_demo_data(days=45, stores=1, products=1))
    features = create_features(raw)
    assert len(raw) == 45
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features.iloc[0].lag_1 != features.iloc[0].lag_1  # first lag is NaN
    assert features.iloc[30].lag_30 == raw.iloc[0].sales


def test_generic_csv_can_be_mapped_to_forecasting_schema():
    raw = make_demo_data(days=10, stores=1, products=1).rename(columns={"date": "timestamp", "sales": "units", "store_id": "region"})
    mapped = adapt_uploaded_schema(raw, "timestamp", "units", "region")
    assert {"date", "store_id", "product_id", "sales"}.issubset(mapped.columns)
