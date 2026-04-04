import os
import sys
import pickle
import logging
from pathlib import Path
import pandas as pd
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import shutil
import json

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add module to path so we can import from scripts
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "scripts"))

try:
    from train_core_predictors import (
        train_return_predictor,
        train_delivery_delay_forecaster,
        train_satisfaction_predictor,
        train_revenue_predictor
    )
except ImportError as e:
    logger.warning(f"Could not import training scripts: {e}")

MODELS_DIR = "models/core_predictors"
models = {}

def load_models():
    """Loads all 4 trained XGBoost pipelines into memory."""
    model_names = [
        "return_predictor.pkl",
        "delay_forecaster.pkl",
        "csat_predictor.pkl",
        "revenue_predictor.pkl"
    ]

    for name in model_names:
        path = os.path.join(base_dir, MODELS_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name.split(".")[0]] = pickle.load(f)
            logger.info(f"Loaded {name}")
        else:
            logger.warning(f"Missing model artifact: {path}")

@asynccontextmanager
async def lifespan(app):
    load_models()
    yield

# Initialize FastAPI
app = FastAPI(
    title="SUPPLYMIND AI", 
    description="Enterprise Decision Intelligence API for Supply Chain Optimization.", 
    version="1.0.0",
    docs_url="/docs",
    redaste_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "SupplyMind Support",
        "url": "https://supplymind.ai/contact",
    }
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_metrics_store = {"total_requests": 0, "total_errors": 0, "total_latency_ms": 0.0}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        api_metrics_store["total_requests"] += 1
        api_metrics_store["total_latency_ms"] += process_time * 1000
    except Exception as e:
        api_metrics_store["total_errors"] += 1
        logger.error(f"Request failed: {e}")
        raise e
    return response

# Input Schema matching the models' exact training features
from pydantic import Field
class OrderContext(BaseModel):
    product_price: float = Field(default=120.0, ge=0.0, description="Price of the product")
    quantity: int = Field(default=1, ge=1, le=1000, description="Quantity ordered")
    discount_percentage: float = Field(default=15.0, ge=0.0, le=100.0, description="Discount applied in percent")
    distance_km: float = Field(default=450.0, ge=0.0, description="Distance from warehouse to customer")
    expected_delivery_days: float = Field(default=3.0, ge=1.0, description="Estimated delivery days")
    past_return_rate: float = Field(default=0.12, ge=0.0, le=1.0, description="Customer past return rate")
    defect_rate: float = Field(default=0.02, ge=0.0, le=1.0, description="Estimated product defect rate")
    product_category: str = Field(default="Electronics", description="Category of the product")
    shipping_mode: str = Field(default="Standard", description="Shipping Mode (e.g. Express, Standard)")
    is_remote_area: int = Field(default=0, ge=0, le=1, description="Whether location is a remote area")
    warehouse_city: str = Field(default="Kolkata", description="Source warehouse city")
    customer_city: str = Field(default="Bangalore", description="Destination city")
    base_return_tendency: float = Field(default=0.10, ge=0.0, le=1.0, description="Baseline return tendency of product")

@app.post("/predict")
def predict_order(order: OrderContext):
    # 1. DELAY FORECASTER
    delay_features = pd.DataFrame([{
        "distance_km": order.distance_km,
        "expected_delivery_days": order.expected_delivery_days,
        "warehouse_city": order.warehouse_city,
        "customer_city": order.customer_city,
        "shipping_mode": order.shipping_mode,
        "is_remote_area": order.is_remote_area
    }])
    if "delay_forecaster" in models:
        delay_pred = float(models["delay_forecaster"].predict(delay_features)[0])
        delay_pred = max(0.0, delay_pred)
    else:
        delay_pred = (order.distance_km / 200.0)
    
    # 2. RETURN PREDICTOR
    return_features = pd.DataFrame([{
        "discount_percentage": order.discount_percentage,
        "distance_km": order.distance_km,
        "past_return_rate": order.past_return_rate,
        "base_return_tendency": order.base_return_tendency,
        "defect_rate": order.defect_rate,
        "delivery_delay": delay_pred,
        "product_category": order.product_category,
        "shipping_mode": order.shipping_mode,
        "is_remote_area": order.is_remote_area
    }])
    if "return_predictor" in models:
        return_prob = float(models["return_predictor"].predict_proba(return_features)[0][1])
        is_returned_pred = int(models["return_predictor"].predict(return_features)[0])
    else:
        return_prob = 0.15
        is_returned_pred = 0

    # 3. CSAT PREDICTOR
    csat_features = pd.DataFrame([{
        "delivery_delay": delay_pred,
        "defect_rate": order.defect_rate,
        "discount_percentage": order.discount_percentage,
        "past_return_rate": order.past_return_rate,
        "product_category": order.product_category,
        "is_returned": is_returned_pred
    }])
    if "csat_predictor" in models:
        csat_pred = float(models["csat_predictor"].predict(csat_features)[0])
        csat_pred = max(1.0, min(5.0, csat_pred))
    else:
        csat_pred = 4.5

    # 4. REVENUE PREDICTOR
    revenue_features = pd.DataFrame([{
        "quantity": order.quantity,
        "product_price": order.product_price,
        "discount_percentage": order.discount_percentage,
        "past_return_rate": order.past_return_rate,
        "product_category": order.product_category,
        "customer_city": order.customer_city
    }])
    if "revenue_predictor" in models:
        rev_pred = float(models["revenue_predictor"].predict(revenue_features)[0])
        rev_pred = max(0.0, rev_pred)
    else:
        rev_pred = order.product_price * order.quantity

    # 5. PRESCRIPTIVE LOGIC
    if return_prob > 0.50 and order.discount_percentage > 20:
        prescription = "⚠️ WARDROBER RISK: Cap discount at 15% for this customer. Do not offer free return shipping."
    elif delay_pred > 3.0:
        prescription = "🚚 LOGISTICS ALERT: High probability of SLA breach. Prescribe upgrade to Express Shipping to prevent CSAT drop."
    elif order.defect_rate > 0.05 and return_prob > 0.30:
        prescription = "🏭 WAREHOUSE ALERT: High defect probability. Flag SKU for pre-shipment Quality Control check."
    else:
        prescription = "✅ OPTIMAL MAPPING: Order clear for standard fulfillment pipeline."

    return {
        "delay_days_estimated": round(delay_pred, 1),
        "return_probability_pct": round(return_prob * 100, 1),
        "csat_projected": round(csat_pred, 1),
        "revenue_projected": round(rev_pred, 2),
        "master_prescription": prescription
    }

@app.post("/predict/batch")
def predict_batch(orders: List[OrderContext]):
    """Batch prediction mode where users can process multiple orders at once."""
    if len(orders) > 1000:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Batch size limit is 1000 orders.")
    results = []
    for o in orders:
        try:
            res = predict_order(o)
            # determine visual risk badge
            risk = "🟢 OK"
            if "WARNING" in res["master_prescription"].upper() or "WARDROBER" in res["master_prescription"].upper() or "ALERT" in res["master_prescription"].upper():
                risk = "🟡 Warning"
                if res["return_probability_pct"] > 60.0 or res["delay_days_estimated"] > 5.0:
                    risk = "🔴 Critical"
            res["risk_badge"] = risk
            results.append(res)
        except Exception as e:
            logger.error(f"Error predicting order: {e}")
            results.append({"error": str(e)})
    return {"status": "success", "results": results}

@app.get("/health")
def health_check():
    """Health check endpoint for frontend API status indicator."""
    loaded = list(models.keys())
    return {
        "status": "ok",
        "models_loaded": len(loaded),
        "models": loaded
    }

@app.get("/insights")
def get_insights():
    """Returns all 4 model insight JSON files for the dashboard."""
    insights_data = {}
    insights_dir = os.path.join(base_dir, "insights")
    files = {
        "return_predictor": "return_predictor_insights.json",
        "delay_forecaster": "delay_forecaster_insights.json",
        "csat_predictor": "csat_predictor_insights.json",
        "revenue_predictor": "revenue_predictor_insights.json"
    }
    for key, fn in files.items():
        path = os.path.join(insights_dir, fn)
        if os.path.exists(path):
            with open(path, "r") as f:
                insights_data[key] = json.load(f)
    return {"status": "ok", "insights": insights_data}

@app.get("/metrics")
def get_api_usage_metrics():
    """Returns API usage statistics for the monitoring dashboard."""
    avg_latency = api_metrics_store["total_latency_ms"] / api_metrics_store["total_requests"] if api_metrics_store["total_requests"] > 0 else 0
    return {
        "total_requests": api_metrics_store["total_requests"],
        "total_errors": api_metrics_store["total_errors"],
        "avg_latency_ms": round(avg_latency, 2),
        "uptime_seconds": round(time.time() - start_time_global, 2)
    }

start_time_global = time.time()

@app.post("/train")
async def train_models_endpoint(file: UploadFile = File(...)):
    """Uploads a CSV dataset, trains the 80/20 models, and returns the insights."""
    try:
        # Save temp file
        temp_path = os.path.join(base_dir, "data", "temp_uploaded_dataset.csv")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load df
        df = pd.read_csv(temp_path)
        logger.info(f"Training started on uploaded dataframe: {df.shape}")

        # Train models
        train_return_predictor(df)
        train_delivery_delay_forecaster(df)
        train_satisfaction_predictor(df)
        train_revenue_predictor(df)

        # Load insights
        insights_data = {}
        insights_dir = os.path.join(base_dir, "insights")
        for fn in ["return_predictor_insights.json", "delay_forecaster_insights.json", "csat_predictor_insights.json", "revenue_predictor_insights.json"]:
            insight_path = os.path.join(insights_dir, fn)
            if os.path.exists(insight_path):
                with open(insight_path, "r") as f:
                    insights_data[fn.split("_insights")[0]] = json.load(f)
        
        # Reload models into memory so predictions work immediately
        load_models()

        return {"status": "success", "message": "Models trained successfully on 80/20 split.", "insights": insights_data}

    except Exception as e:
        logger.error(f"Error training models: {e}")
        return {"status": "error", "message": str(e)}

# --- Serve Frontend ---
# Catch-all: serve the SPA index.html for the root path
web_dir = os.path.join(base_dir, "web")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(web_dir, "index.html"))

# Mount static assets (if any CSS/JS/images are added later)
if os.path.isdir(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

