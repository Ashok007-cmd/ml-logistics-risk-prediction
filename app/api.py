"""
FastAPI backend for Logistics Risk Prediction.

Serves the ML model via a robust HTTP API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predictor import Predictor

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global predictor
    # Load model into memory once on startup
    predictor = Predictor()
    yield
    predictor = None

app = FastAPI(
    title="Logistics Risk Prediction API",
    description="API for predicting late-delivery risks for shipments.",
    version="1.0.0",
    lifespan=lifespan,
)

class PredictionRequest(BaseModel):
    # Defining the fields expected by the model
    shipping_mode: str = Field(..., alias="Shipping Mode")
    customer_segment: str = Field(..., alias="Customer Segment")
    customer_city: str = Field("New York", alias="Customer City")
    customer_state: str = Field("NY", alias="Customer State")
    customer_country: str = Field("United States", alias="Customer Country")
    market: str = Field(..., alias="Market")
    order_region: str = Field(..., alias="Order Region")
    product_category_name: str = Field(..., alias="Product Category Name")
    department_name: str = Field(..., alias="Department Name")
    type: str = Field("DEBIT", alias="Type")
    product_name: str = Field("Unknown Product", alias="Product Name")
    customer_full_name: str = Field("Unknown Customer", alias="Customer Full Name")
    order_city: str = Field("Unknown City", alias="Order City")
    days_for_shipment_scheduled: int = Field(
        ..., alias="Days for shipment (scheduled)", ge=0, le=90
    )
    latitude: float = Field(..., alias="Latitude", ge=-90, le=90)
    longitude: float = Field(..., alias="Longitude", ge=-180, le=180)
    order_item_discount: float = Field(..., alias="Order Item Discount", ge=0)
    order_item_product_price: float = Field(..., alias="Order Item Product Price", ge=0)
    order_item_quantity: int = Field(..., alias="Order Item Quantity", gt=0)
    product_price: float = Field(..., alias="Product Price", ge=0)
    order_item_total: float = Field(..., alias="Order Item Total", ge=0)
    order_profit_per_order: float = Field(0.0, alias="Order Profit Per Order")
    sales_per_customer: float = Field(..., alias="Sales per customer", ge=0)
    product_card_id: int = Field(0, alias="Product Card Id", ge=0)
    customer_zipcode: int = Field(10001, alias="Customer Zipcode", ge=0)
    order_date: str = Field(..., alias="order date (DateOrders)")

    model_config = ConfigDict(populate_by_name=True)

class FeatureContribution(BaseModel):
    feature: str
    contribution: float

class PredictionResponse(BaseModel):
    probability: float
    risk_label: str
    threshold_used: float
    contributions: list[FeatureContribution]

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Convert Pydantic model back to a dict mimicking the expected row
    data = request.model_dump(by_alias=True)
    df = pd.DataFrame([data])

    prob, risk = predictor.risk_score(df)
    contrib_df = predictor.feature_contributions(df, top_k=8)

    contributions = [
        FeatureContribution(feature=row["feature"], contribution=row["contribution"])
        for _, row in contrib_df.iterrows()
    ]

    return PredictionResponse(
        probability=prob,
        risk_label=risk,
        threshold_used=predictor.threshold["threshold"],
        contributions=contributions,
    )

@app.get("/health")
def health_check() -> dict[str, str]:
    if predictor is None:
        return {"status": "starting"}
    return {"status": "ok"}
