from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "models_loaded" in response.json()

def test_predict_single_order():
    payload = {
        "product_price": 100.0,
        "quantity": 2,
        "discount_percentage": 5.0,
        "distance_km": 200.0,
        "expected_delivery_days": 2.0,
        "past_return_rate": 0.05,
        "defect_rate": 0.01,
        "product_category": "Electronics",
        "shipping_mode": "Standard",
        "is_remote_area": 0,
        "warehouse_city": "Kolkata",
        "customer_city": "Bangalore",
        "base_return_tendency": 0.1
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "delay_days_estimated" in data
    assert "return_probability_pct" in data
    assert "csat_projected" in data
    assert "revenue_projected" in data
    assert "master_prescription" in data
    # Bounds checking
    assert data["return_probability_pct"] >= 0.0 and data["return_probability_pct"] <= 100.0
    assert data["csat_projected"] >= 1.0 and data["csat_projected"] <= 5.0
    assert data["revenue_projected"] >= 0.0

def test_predict_batch():
    payload = [
        {
            "product_price": 100.0,
            "quantity": 2,
            "discount_percentage": 5.0,
            "distance_km": 200.0,
            "expected_delivery_days": 2.0,
            "past_return_rate": 0.05,
            "defect_rate": 0.01,
            "product_category": "Electronics",
            "shipping_mode": "Standard",
            "is_remote_area": 0,
            "warehouse_city": "Kolkata",
            "customer_city": "Bangalore",
            "base_return_tendency": 0.1
        },
        {
             "product_price": 500.0,
             "quantity": 1,
             "discount_percentage": 25.0,
             "distance_km": 1500.0,
             "expected_delivery_days": 6.0,
             "past_return_rate": 0.40,
             "defect_rate": 0.05,
             "product_category": "Home",
             "shipping_mode": "Express",
             "is_remote_area": 1,
             "warehouse_city": "Delhi",
             "customer_city": "Mumbai",
             "base_return_tendency": 0.3
        }
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["results"]) == 2
    assert "risk_badge" in data["results"][0]

def test_predict_invalid_payload():
    # Only sending price is invalid because of required fields in OrderContext
    payload = {
        "product_price": -50.0 
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
