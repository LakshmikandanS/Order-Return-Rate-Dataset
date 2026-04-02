# 🚀 Advanced Preprocessing & Feature Engineering Plan (Detailed)

This document outlines a high-granularity preprocessing pipeline to transform the current synthetic dataset—or any e-commerce dataset—into a high-quality format optimized for predicting order returns.

---

## 🏗️ 1. Pipeline Architecture
The pipeline is divided into five distinct phases:
1. **Data Integrity & Structural Cleaning**
2. **Temporal & Seasonal Engineering**
3. **Behavioral & Historical Profiling**
4. **NLP & Sentiment extraction (Review Intelligence)**
5. **Model-Ready Transformation (Encoding & Scaling)**

---

## 🛠️ 2. Detailed Phase Breakdown

### Phase 1: Data Integrity & Cleaning
- **Missing Value Strategy**:
  - `review_text`, `review_rating`, `review_date`: Approximately 35% missingness.
    - *Policy*: Do not drop. Create a boolean flag `has_review`. Impute `review_rating` with 0 (neutral/none) for predictive purposes if using non-text models.
  - `return_reason`: For `is_returned=0`, this is `NO_RETURN`. Convert to a categorical placeholder.
- **Outlier Management (Robustness)**:
  - `product_price`: Use robust scaling or log-transform if skewness > 2.0.
  - `distance_km`: Check for unrealistic spikes (e.g., > 3000km in a domestic context).
- **Recalculation & Validation**:
  - Recalculate `order_value = product_price * quantity` to verify integrity.
  - Ensure `actual_delivery_days >= 0`. Fix or flag negative delays.

### Phase 2: Temporal & Seasonal Engineering
- **Cyclical Features**: Extract `hour`, `day_of_week`, `month`, and `is_weekend` from `order_date`.
- **Festival Awareness**:
  - `days_to_nearest_festival`: Calculate distance to major spikes (Diwali, Christmas).
  - `is_festival_window`: Binary flag if within +/- 7 days of a major event.
- **Processing Lag (Operational Efficiency)**:
  - `review_lag`: Time difference between `order_date` and `review_date`. High lag often correlates with return processing.

### Phase 3: Behavioral & Historical Profiling
- **Customer Rolling Features (CRF)**:
  - `cust_past_3_returns`: Count of returns in the last 30 days.
  - `cust_lifetime_value`: Cumulative `order_value`.
  - `cust_avg_rating`: Running average of ratings given by this customer.
- **Product Risk Metrics (PRM)**:
  - `prod_return_rate_3t`: Rolling return rate for this product category over the last 3 orders.
  - `is_high_defect_prod`: Binary flag if product `defect_rate` > threshold (e.g., 0.1).

### Phase 4: Review Intelligence (NLP)
- **Sentiment Scoring**:
  - Use VADER or a similar lightweight analyzer to get a `sentiment_polarity` score.
- **Sarcasm Detection (Heuristic)**:
  - Flag comments containing "matches my cat", "excellent quality control", "horrible", etc.
  - Feature: `is_sarcastic` (boolean).
- **Text Vectorization**:
  - TF-IDF or Word Embeddings for `review_text` to capture themes like "fit issue" or "color mismatch".

### Phase 5: Model-Ready Transformation
- **Categorical Encoding**:
  - `product_category`, `customer_city`, `warehouse_city`: Use Target Encoding (with smoothing).
  - `shipping_mode`: One-hot encoding.
- **Scaling & Normalization**:
  - `StandardScaler` for normally distributed features.
  - `MinMaxScaler` for skewed features like `distance_km`.
- **Leakage Prevention**:
  - **CRITICAL**: Ensure all historical/rolling features are calculated using time-based splits to avoid look-ahead bias.

---

## ⚙️ 3. Preprocessing Logic Table

| Feature | Transformation | Reasoning |
| :--- | :--- | :--- |
| `order_date` | Extract Cyclical + Festival | Captures seasonal/weekend return behaviors. |
| `review_text` | TF-IDF + Sentiment Score | Captures qualitative dissatisfaction (sarcasm/anger). |
| `distance_km` | Binning or Log-Scaling | Long distance increases damage risk (Quality Defect). |
| `delivery_delay` | Poly Features (`delay^2`) | Risk of return increases exponentially with delay. |
| `is_remote_area` | Interaction with `shipping_mode` | Remote + Express often leads to higher delay. |

---

## 🚀 4. Implementation Checklist
- [ ] Create `preprocess.py` script.
- [ ] Implement `handle_missing_values()` function.
- [ ] Implement `extract_temporal_features()` function.
- [ ] Implement `calculate_rolling_behavior()` function.
- [ ] Run a baseline `XGBoost` or `RandomForest` to validate feature importance.

---

*Last Updated: April 2, 2026*

# 🚀 Adaptive ML Pipeline for Reverse Logistics (LEGACY)
# ...existing code...


---

## 3.3 Outlier Handling

* Detect extreme values (IQR / percentile method)
* Apply:

  * Clipping
  * Winsorization

---

## 3.4 Feature Scaling

* Apply scaling where needed:

  * Standardization (Z-score)
  * Min-Max scaling

*(Important for linear models; optional for tree models)*

---

## 3.5 Temporal & Cyclical Encoding

* **Date Components**: Extract `month`, `day_of_week`, `quarter` from timestamps (e.g., `order_date`).
* **Sine/Cosine Transformations**: Encode cyclical features (like month or day of week) mapping month 12 adjacent to month 1.

---

## 3.6 Textual Data & NLP (Reviews) 🧠 NEW

* **Text Cleaning**: Lowercasing, punctuation removal, and stopword filtering for raw text (e.g., `review_text`).
* **Text Representation**:
  * Generate basic metrics (e.g., `review_length`, `word_count`).
  * Implement NLP embeddings: TF-IDF, fastText, or pre-trained Sentence Transformers.
* **Sentiment Analysis**:
  * Extract polarity and subjectivity scores using libraries like VADER or pre-trained HuggingFace BERT models.

---

## 3.7 Target Encoding for Latent Entities

In real-world data, latent properties (e.g., `base_return_tendency` and `defect_rate`) are not handed out explicitly.
* **Historical Smoothing**: Instead of raw target encoding (which risks data leakage), compute **rolling historical aggregate functions** (e.g., expanding average return rate using past $N$ orders up to `time = t - 1`).
* **K-Fold Target Encoding**: When rolling windows are unavailable, use K-Fold target encoding for `product_id` or `customer_id` strictly outside the test set.

---

# 🔗 4. Multicollinearity Analysis

## 4.1 Definition

Multicollinearity occurs when:

> Multiple input features are highly correlated with each other

---

## 4.2 Sources in E-commerce Data

* Derived relationships:

  * Order value ↔ price & quantity
  * Discount amount ↔ discount percentage
  * Delivery delay ↔ delivery days

---

## 4.3 Detection Methods

### Correlation Matrix

* Identify highly correlated feature pairs

### Variance Inflation Factor (VIF)

* Quantify multicollinearity
* Thresholds:

  * VIF > 10 → High
  * VIF 5–10 → Moderate

---

## 4.4 Mitigation Strategies

* Remove redundant features
* Retain only one from correlated groups
* Prefer **tree-based models** (robust to multicollinearity)

---

# 📊 5. Data Validation & Quality Checks

## 5.1 Logical Consistency

* Order value correctness
* Delivery relationships
* Discount calculations

---

## 5.2 Statistical Validation

* Return rate within realistic bounds (10–40%)
* Distribution checks:

  * Price (right-skewed)
  * Quantity (low integer values)

---

## 5.3 Behavioral Validation

* Delay positively impacts returns
* High discount increases returns
* Clothing category shows higher return rate

---

# 🔄 6. Handling Missing Features

## 6.1 Strategy

| Feature Type | Handling            |
| ------------ | ------------------- |
| Required     | Reject dataset      |
| Derivable    | Compute dynamically |
| Optional     | Fill / ignore       |
| Behavioral   | Create `has_feature` indicator |

---

## 6.2 Fallback Mechanisms

* Default values for missing optional features
* Reduced feature set for prediction
* Adaptive model selection

---

## 6.3 Behavioral Missingness (e.g., Reviews)

* When a customer doesn't leave a `review_rating` or `review_text`, the missing data is **informative**.
* Do **not** impute with typical mode/mean.
* Instead, generate an explicit binary feature `is_reviewed`.
* For numerical features like `review_rating`, impute with `0` or an out-of-scale number (-1). For NLP pipelines, map missing texts to an empty string `""` or `"NO_REVIEW"`.

---

# 🚫 7. Data Leakage Control

## 7.1 Temporal Considerations

* **Predicting Point-of-Sale Returns**: At the moment a customer buys, `delivery_delay`, `review_text`, and `actual_delivery_days` **do not exist**.
* Including them here will cause 100% data leakage!

## 7.2 Post-Purchase / Diagnostic Modeling

* If the goal is **Risk Analysis after Delivery**, it is perfectly valid to use `delivery_delay` and `review_rating`.
* Maintain two feature sets (Pre-Purchase vs. Post-Purchase) strictly managed via configuration files.
* Remove `is_returned` logic from standard modeling pipelines immediately before the `train_test_split()`.

---

# ⚠️ 8. Distribution Shift Handling

## 8.1 Problem

User-uploaded data may differ from training data:

* Different scales
* Different customer behavior
* Different distributions

---

## 7.2 Detection

* Compare:

  * Mean
  * Standard deviation
  * Distribution shape

---

## 7.3 Response

* Re-scaling
* Feature normalization
* Model retraining if shift is significant

---

# 🧩 8. Model Robustness Strategy

## 8.1 Feature Dependency Control

* Avoid reliance on all features
* Focus on high-importance features

---

## 8.2 Model Choice

* Tree-based models preferred:

  * Handle missing values
  * Robust to multicollinearity
  * Capture non-linear relationships

---

# 🔍 9. Customer-Level Insight Engine

System supports queries like:

> “Analyze customer behavior”

Outputs:

* Return probability
* Behavioral patterns
* Risk segmentation

---

## Example Insights

* High return tendency in specific categories
* Sensitivity to discounts
* Impact of delivery delays

---

# 🚀 10. End-to-End Pipeline Flow

```
User Upload
    ↓
Schema Detection
    ↓
Feature Mapping & Derivation
    ↓
Data Cleaning & Preprocessing
    ↓
Multicollinearity Check
    ↓
Validation & Consistency Checks
    ↓
Distribution Analysis
    ↓
Model Selection / Adaptation
    ↓
Prediction & Insights
```

---

# 🧾 11. Key Design Principles

* Schema-aware, not schema-dependent
* Robust to missing and inconsistent data
* Causality-preserving transformations
* Validation-driven processing
* Model adaptability

---

# 🎯 Conclusion

This pipeline ensures:

* Reliable predictions across varying datasets
* Strong data integrity and validation
* Real-world applicability in e-commerce systems

It transforms a rigid ML model into:

> **An adaptive, resilient decision-support system for reverse logistics**

---
