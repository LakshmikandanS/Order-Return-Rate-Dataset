# SUPPLYMIND: Enterprise AI Decision Intelligence

## 🚀 Business Potential & ROI Analysis

SUPPLYMIND transforms standard e-commerce logistics from a **reactive** (waiting for returns) to a **proactive** model. By predicting returns, delays, and CSAT *before* a single parcel is shipped, businesses can realize significant savings and customer retention.

### 💰 Key ROI Metrics
| Pillar | Logic | Potential Savings |
|---|---|---|
| **Return Mitigation** | 50% accurate return prediction → 20% reduction in reverse logistics. | $150k - $500k / yr |
| **Delay Prevention** | Proactive warning on late shipping allows rerouting. | 15% Lower Churn |
| **CSAT Optimization** | Predicting low satisfaction allows customer support to issue early credits. | 30% Higher LTV |

### 🛠 Tech Stack Highlights
- **ML Engine**: XGBoost Classifiers & Regressors with Scale-Pos-Weight for rare-event (return) detection.
- **Backend**: FastAPI (Python) - High performance, asynchronous inference.
- **Frontend**: Glassmorphism Dashboard (Vanilla JS/CSS) for executive visibility.
- **Port Strategy**:
    - `8001`: Core AI REST API (Inference Engine).
    - `8000`: Executive Dashboard Proxy.

### 🔌 Integration & Extension
The SUPPLYMIND API is designed to be injected into existing ERP systems (SAP, Oracle, Odoo). It consumes `JSON` order objects and returns a `Master Prescription` within **<50ms**.

---

## 📖 User Operations Manual

### 1. Launching the Platform
1.  **Backend AI**: `cd api && python main.py` (Runs on Port 8001).
2.  **Dashboard**: `cd web && python -m http.server 8000` (Runs on Port 8000).

### 2. Using the Dashboard
- **Simulation Input**: Fill the sidebar fields (Price, Discount, Past Returns).
- **Run Prediction**: Click the **Run Inference** button.
- **Master Prescription**: Observe the "Business Logic Header" (e.g., "⚠️ WARDROBER RISK" or "✅ OPTIMAL MAPPING").

### 3. API Documentation
**POST** `http://localhost:8001/predict`
```json
{
  "order_item_id": 101,
  "price": 299.99,
  "freight_value": 15.0,
  "product_category": "electronics",
  "customer_state": "NY",
  "total_discount": 0.4,
  "is_installment": 0,
  "customer_past_return_rate": 0.8,
  "customer_order_frequency": 12,
  "carrier_id": 1,
  "distance_km": 450
}
```
