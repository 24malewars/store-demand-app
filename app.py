import re
import joblib
import glob
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Store Demand Forecast", page_icon="📦", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


try:
    model, scaler, encoder = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# Pull the valid store / item names straight from the fitted encoder
cat_enc = encoder.named_transformers_["cat"].named_steps["encoder"]
stores = sorted(cat_enc.categories_[0], key=natural_key)
items = sorted(cat_enc.categories_[1], key=natural_key)

st.title("📦 Next-Month Demand Forecast")
st.caption("Predict units sold for a store-item pair next month (Random Forest model).")

col1, col2 = st.columns(2)
store = col1.selectbox("Store", stores)
item = col2.selectbox("Item", items)

col3, col4 = st.columns(2)
year = col3.number_input("Current year", min_value=2015, max_value=2100, value=2022, step=1)
month_number = col4.selectbox("Current month", list(range(1, 13)), index=6,
                              format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))

st.subheader("Past monthly sales (units)")
c1, c2, c3 = st.columns(3)
lag_1 = c1.number_input("lag_1 (1 month before)", min_value=0.0, value=1000.0, step=10.0)
lag_2 = c2.number_input("lag_2 (2 months before)", min_value=0.0, value=1000.0, step=10.0)
lag_3 = c3.number_input("lag_3 (3 months before)", min_value=0.0, value=1000.0, step=10.0)

rolling_3 = (lag_1 + lag_2 + lag_3) / 3
st.info(f"rolling_3 (average of the 3 lags) = **{rolling_3:,.1f}**")

if st.button("Predict", type="primary", use_container_width=True):
    # Same column names and order as used in training
    row = pd.DataFrame([{
        "store": store, "item": item, "year": year, "month_number": month_number,
        "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "rolling_3": rolling_3,
    }])
    encoded = encoder.transform(row)
    scaled = scaler.transform(encoded)
    pred = float(model.predict(scaled)[0])

    st.metric("Predicted units next month", f"{pred:,.0f}")
    chart = pd.DataFrame(
        {"Units": [lag_3, lag_2, lag_1, pred]},
        index=["lag_3", "lag_2", "lag_1", "Prediction"],
    )
    st.bar_chart(chart)
