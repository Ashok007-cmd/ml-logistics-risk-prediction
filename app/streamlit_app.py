"""
Streamlit deployment app for logistics risk prediction.

Loads the serialized XGBoost model + preprocessing pipeline and provides
an interactive form for ops staff to input shipment parameters and receive
instant risk assessments with feature contribution breakdowns.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Logistics Risk Predictor",
    page_icon="🚚",
    layout="centered",
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/predict"
REQUEST_TIMEOUT_SECONDS = 10

st.title("Logistics Risk Predictor")
st.markdown("Enter shipment details below to get an instant **late-delivery risk assessment**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ["Standard Class", "First Class", "Second Class", "Same Day"],
        )
        customer_segment = st.selectbox(
            "Customer Segment",
            ["Consumer", "Corporate", "Home Office"],
        )
        market = st.selectbox(
            "Market",
            ["US", "LATAM", "Europe", "Pacific", "Africa"],
        )
        order_region = st.selectbox(
            "Order Region",
            ["North", "South", "East", "West", "Central"],
        )
        department = st.selectbox(
            "Department Name",
            ["Furniture", "Office Supplies", "Technology"],
        )
        category = st.selectbox(
            "Product Category Name",
            [
                "Office Machines",
                "Chairs",
                "Tables",
                "Phones",
                "Storage",
                "Art",
                "Binders",
                "Paper",
                "Appliances",
            ],
        )

    with col2:
        scheduled_days = st.number_input(
            "Days for shipment (scheduled)",
            min_value=0,
            max_value=30,
            value=4,
        )
        discount = st.number_input(
            "Order Item Discount",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        quantity = st.number_input(
            "Order Item Quantity",
            min_value=1,
            max_value=100,
            value=2,
        )
        product_price = st.number_input(
            "Product Price ($)",
            min_value=0.0,
            max_value=10000.0,
            value=150.0,
            step=10.0,
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=40.7128,
        )
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-74.0060,
        )

    submitted = st.form_submit_button("Predict Risk", type="primary")

if submitted:
    order_date_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    payload = {
        "Shipping Mode": shipping_mode,
        "Customer Segment": customer_segment,
        "Customer City": "New York",
        "Customer State": "NY",
        "Customer Country": "United States",
        "Market": market,
        "Order Region": order_region,
        "Product Category Name": category,
        "Department Name": department,
        "Type": "DEBIT",
        "Product Name": "Unknown Product",
        "Customer Full Name": "Unknown Customer",
        "Order City": "Unknown City",
        "Days for shipment (scheduled)": int(scheduled_days),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
        "Order Item Discount": float(discount),
        "Order Item Product Price": float(product_price),
        "Order Item Quantity": int(quantity),
        "Product Price": float(product_price),
        "Order Item Total": float(product_price) * int(quantity) * (1 - float(discount)),
        "Order Profit Per Order": 0.0,
        "Sales per customer": float(product_price) * int(quantity),
        "Product Card Id": 0,
        "Customer Zipcode": 10001,
        "order date (DateOrders)": order_date_str,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        result = response.json()

        prob = result["probability"]
        risk = result["risk_label"]
        threshold_value = result["threshold_used"]
        contributions = result["contributions"]

        st.divider()

        if risk == "HIGH RISK":
            st.markdown("### Risk Assessment: :red[HIGH RISK]")
        else:
            st.markdown("### Risk Assessment: :green[LOW RISK]")

        col1, col2, col3 = st.columns(3)
        col1.metric("Late-Delivery Probability", f"{prob:.1%}")
        col2.metric("Decision Threshold", f"{threshold_value:.0%}")
        col3.metric("Confidence", "High" if abs(prob - 0.5) > 0.2 else "Medium")

        if risk == "HIGH RISK":
            st.warning(
                f"This shipment has a **{prob:.1%}** probability of late delivery. "
                f"Consider escalating or selecting an alternative carrier."
            )
        else:
            st.success(
                f"This shipment has a **{prob:.1%}** probability of late delivery. "
                f"No action required."
            )

        with st.expander("Feature Contributions (what drove this prediction)"):
            contrib_df = pd.DataFrame(contributions)
            st.bar_chart(contrib_df.set_index("feature")["contribution"])
            st.caption("Positive values push toward late-delivery risk; negative values push away.")

        with st.expander("Prediction Details"):
            st.json(
                {
                    "probability": round(prob, 4),
                    "threshold": threshold_value,
                    "risk_label": risk,
                    "shipping_mode": shipping_mode,
                    "scheduled_days": scheduled_days,
                }
            )
    except requests.exceptions.Timeout:
        st.error(
            f"Backend at {API_URL} did not respond within "
            f"{REQUEST_TIMEOUT_SECONDS}s. Is the FastAPI service running?"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch prediction from backend: {e}")
