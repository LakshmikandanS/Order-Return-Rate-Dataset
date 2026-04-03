# 🚀 Pipeline Improvement Plan (Post Olist Evaluation)

**Date**: April 3, 2026  
**Context**: Improvements derived from external dataset testing on Olist (Real Mode)

---

## 🎯 Objective
Refine the preprocessing and modeling pipeline to:
- Improve real-world generalization
- Eliminate weak or misleading features
- Adapt dynamically to dataset characteristics
- Transition from synthetic assumptions → real data intelligence

---

## ⚠️ Key Issues Identified

### 1. Spam Detection Overestimation
- Detected ~57.9% spam (unrealistic)
- Caused by:
  - Language mismatch (Portuguese vs English model)
  - Over-penalizing short reviews

### 2. Underutilization of Real Logistics Data
- Synthetic placeholders used despite availability of real delivery timestamps

### 3. Proxy Target Limitation
- “Return” defined as:
  - `rating ≤ 2 OR sentiment < -0.2`
- Not actual return data → limits realism

### 4. NLP Generalization Weakness
- TextBlob fails for multilingual datasets
- Sarcasm detection drops significantly in non-English data

### 5. Behavioral Features Reliability
- Sparse customer history reduces effectiveness of rolling features

---

## 🔧 Improvement Roadmap

---

## 🔥 Phase 1: Real Feature Utilization (High Priority)

### ✅ Replace Synthetic Logistics with Real Signals

**Features to Add:**
- `delivery_delay = actual_delivery_date - estimated_delivery_date`
- `delivery_speed_ratio = actual / expected`
- `is_late_delivery = (delivery_delay > 0)`

**Impact:**
- Converts pipeline from synthetic → real-world valid
- Strong predictive power for customer dissatisfaction

---

## 🌍 Phase 2: Multilingual NLP Upgrade

### ❌ Current:
- TextBlob (English-only bias)

### ✅ Upgrade To:
- HuggingFace multilingual models:
  - `nlptown/bert-base-multilingual-uncased-sentiment`

### Improvements:
- Accurate sentiment for Portuguese/other languages
- Better sarcasm signal foundation

---

## ⚠️ Phase 3: Spam Detection Recalibration

### Issues:
- Over-detection due to:
  - short text penalty
  - sentiment misinterpretation

### Fix Strategy:
- Reduce weight of:
  - `review_word_count`
- Add:
  - language-aware filtering
- Rebalance thresholds:
  - spam threshold ↑ (e.g., 0.6 → 0.8)

### Goal:
- Reduce spam detection to realistic range (~5–15%)

---

## 🧠 Phase 4: Target Redefinition

### ❌ Current:
- Artificial "return" label

### ✅ New Targets:

#### Option A (Recommended)
- `low_rating = review_score ≤ 2`

#### Option B
- Regression:
  - Predict `review_score`

#### Option C (Advanced)
- Multi-class classification:
  - Positive / Neutral / Negative

### Outcome:
- Aligns problem with real-world measurable behavior

---

## 📊 Phase 5: Feature Validation & Selection

### Actions:
- Remove redundant features:
  - e.g., `order_value` vs `price × quantity`
- Perform:
  - Correlation analysis
  - SHAP value analysis

### Goal:
- Identify truly impactful features
- Reduce noise

---

## 🧪 Phase 6: Model Evaluation (Critical)

### Required:
- Train model on Olist dataset

### Metrics:
- Accuracy
- F1-score
- Confusion Matrix

### Compare Across Datasets:

| Dataset | Expected Behavior |
|--------|-----------------|
| Synthetic | Overfit (100%) |
| Amazon | NLP-driven |
| Olist | Logistics-driven |

---

## 🔄 Phase 7: Adaptive Pipeline Design (Advanced)

### Current Limitation:
- Same logic applied to all datasets

### Upgrade:
- Detect dataset characteristics:
  - Language
  - Feature availability
  - Data density

### Dynamically adjust:
- NLP model selection
- Feature generation
- Thresholds

---

## 🏗️ Phase 8: Architecture Enhancements

### Add:
- Config-driven pipeline:
```python
mode = {
    "use_nlp": True,
    "use_logistics": True,
    "language": "auto"
}