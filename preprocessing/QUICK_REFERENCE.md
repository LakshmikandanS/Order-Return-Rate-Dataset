# 🚀 Preprocessing Script - Quick Reference Guide

## Usage

### Basic Usage
```python
from preprocessing.preprocess import run_preprocessing_pipeline, validate_with_xgboost

# Run complete pipeline
df_processed = run_preprocessing_pipeline(
    input_csv='Order-Return-Rate-Dataset/data/synthetic_ecommerce_orders.csv',
    output_csv='Order-Return-Rate-Dataset/data/synthetic_ecommerce_orders_preprocessed.csv'
)

# Validate with XGBoost
model, feature_importance = validate_with_xgboost(df_processed)
```

### From Command Line
```bash
cd d:\coding_stuffs\minor_project_supply_chain
python preprocessing/preprocess.py
```

---

## Available Functions

### Phase 1: Data Cleaning
```python
from preprocessing.preprocess import (
    handle_missing_values,
    validate_order_integrity,
    handle_outliers
)

df = handle_missing_values(df)           # Handle NaN values
df = validate_order_integrity(df)        # Check data consistency
df = handle_outliers(df)                 # Robust scaling & flagging
```

### Phase 2: Temporal Features
```python
from preprocessing.preprocess import extract_temporal_features

df = extract_temporal_features(df)
# Creates: order_hour, order_day_of_week, order_month, is_weekend,
#          order_quarter, days_to_nearest_festival, is_festival_window,
#          review_lag_days
```

### Phase 3: Behavioral Features
```python
from preprocessing.preprocess import calculate_rolling_behavior

df = calculate_rolling_behavior(df)
# Creates: cust_lifetime_orders, cust_lifetime_value, cust_past_30_returns,
#          cust_avg_rating, prod_category_return_rate, is_high_risk_product
```

### Phase 4: NLP & Sentiment
```python
from preprocessing.preprocess import extract_sentiment_features, vectorize_reviews

df = extract_sentiment_features(df)
# Creates: sentiment_polarity, sentiment_subjectivity, is_sarcastic,
#          review_word_count

df = vectorize_reviews(df, max_features=20)  # Optional TF-IDF
# Creates: tfidf_* features
```

### Phase 5: Encoding & Scaling
```python
from preprocessing.preprocess import encode_categorical_features, scale_numerical_features

# Target encode categorical features
categorical_cols = ['product_category', 'shipping_mode', 'return_reason', 
                   'warehouse_city', 'customer_city']
df = encode_categorical_features(df, categorical_cols, target_col='is_returned')

# Scale numerical features
numerical_cols = ['product_price_log', 'quantity', 'distance_km', ...]
df = scale_numerical_features(df, numerical_cols, method='standard')
```

### Validation
```python
from preprocessing.preprocess import validate_with_xgboost

model, feature_importance = validate_with_xgboost(
    df,
    test_size=0.2,
    random_state=42
)
# Generates: feature_importance.csv with rankings
```

---

## Input/Output Specifications

### Input CSV Structure
**File**: `Order-Return-Rate-Dataset/data/synthetic_ecommerce_orders.csv`

**Required Columns**:
```
order_id, order_date, customer_id, product_id, product_category,
product_price, quantity, order_value, discount_percentage, discount_amount,
customer_city, warehouse_city, distance_km, is_remote_area, shipping_mode,
expected_delivery_days, actual_delivery_days, delivery_delay, is_returned,
return_reason, total_orders, past_return_rate, avg_order_value,
review_rating, review_text, review_date, base_return_tendency,
home_city, defect_rate
```

**Data Types**:
- Numeric: product_price, quantity, order_value, distance_km, delivery_delay, review_rating
- Categorical: product_category, shipping_mode, return_reason, warehouse_city, customer_city
- DateTime: order_date, review_date
- Boolean: is_returned, is_remote_area
- Text: review_text, return_reason

### Output CSV Structure
**File**: `Order-Return-Rate-Dataset/data/synthetic_ecommerce_orders_preprocessed.csv`

**Columns**: 69 total (29 original + 40 engineered)

**Categories**:
1. Original features (29) - all preserved
2. Quality indicators (1) - `has_review`
3. Temporal features (9) - time decomposition + lag
4. Behavioral features (4) - customer history
5. Risk metrics (2) - product-level risk
6. Sentiment features (5) - NLP analysis
7. Derived features (18) - scaled versions of numericals
8. Encoded features (10) - target-encoded categoricals
9. Other (1) - outlier flags

---

## Configuration Options

### Missing Value Strategy
```python
# Current: Impute review_rating with 0, review_text with 'NO_REVIEW'
# Modify in handle_missing_values():
df['review_rating'] = df['review_rating'].fillna(CUSTOM_VALUE)
```

### Outlier Thresholds
```python
# Current: Log-transform if skewness > 2.0
# Modify in handle_outliers():
SKEWNESS_THRESHOLD = 2.0
DISTANCE_THRESHOLD = 3000  # km
```

### Festival Dates
```python
# Current: Indian festivals (Diwali, Christmas, etc.)
# Modify in extract_temporal_features():
festival_dates = {
    'Diwali': (10, 25),
    'Christmas': (12, 25),
    # Add/modify as needed
}
```

### Risk Thresholds
```python
# Current: Product flagged if return_rate > 15%
# Modify in calculate_rolling_behavior():
defect_threshold = 0.15
```

### Sarcasm Keywords
```python
# Current: 'matches my cat', 'excellent quality control', 'horrible', etc.
# Modify in extract_sentiment_features():
sarcasm_keywords = [
    'keyword1', 'keyword2', ...
]
```

### Scaling Method
```python
# Options: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
df = scale_numerical_features(df, numerical_cols, method='standard')
```

---

## Performance Characteristics

| Phase | Duration | Complexity | Scalability |
|-------|----------|-----------|-------------|
| Phase 1 | ~20ms | O(n) | Excellent |
| Phase 2 | ~300ms | O(n) | Excellent |
| Phase 3 | ~7.4s | O(n²) worst case | Good* |
| Phase 4 | ~700ms | O(n) tokenization | Excellent |
| Phase 5 | ~80ms | O(n) | Excellent |
| **Total** | **~8.6s** | **O(n²)** | **Good** |

*Phase 3 uses nested loops for time-based windows - can optimize with vectorization for large datasets (>1M rows)

---

## Performance Tips

### For Large Datasets (>100K rows)

1. **Optimize Phase 3**
   ```python
   # Replace loop-based rolling calculations with vectorized operations
   # Use pandas groupby with shifted operations instead of row-by-row iteration
   ```

2. **Parallel Processing**
   ```python
   # Use multiprocessing for Phase 4 (NLP processing)
   # from multiprocessing import Pool
   ```

3. **Memory Optimization**
   ```python
   # Process in chunks if data exceeds system RAM
   chunk_size = 10000
   df_chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
   ```

### Estimated Time for Different Dataset Sizes
```
5,000 records:      8.6 seconds
50,000 records:     ~80 seconds
100,000 records:    ~160 seconds (bottleneck: Phase 3)
500,000 records:    ~800 seconds (~13 minutes)
1,000,000 records:  ~26 minutes (recommend optimization)
```

---

## Troubleshooting

### Missing Module Errors
```bash
pip install pandas numpy scikit-learn xgboost textblob
```

### Memory Limit Exceeded
- Reduce dataset size or process in chunks
- Disable TF-IDF vectorization (`vectorize_reviews()`)
- Clear intermediate dataframes

### Date Parsing Issues
```python
# Ensure date columns are in standard format (YYYY-MM-DD)
df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d')
```

### NaN Propagation
```python
# Some scaled features may have NaN from missing inputs
# Script auto-fills with column mean - check if acceptable
df[col] = df[col].fillna(df[col].mean())
```

---

## Expected Output Logs

```
2026-04-02 15:53:20 - INFO - 🚀 STARTING PREPROCESSING PIPELINE
2026-04-02 15:53:20 - INFO - 📂 Loading data from: ...
2026-04-02 15:53:20 - INFO - 🧹 Phase 1: Handling Missing Values...
2026-04-02 15:53:20 - INFO - 🧹 Phase 2: Extracting Temporal Features...
2026-04-02 15:53:28 - INFO - 👥 Phase 3: Calculating Behavioral Features...
2026-04-02 15:53:28 - INFO - 📝 Phase 4: Extracting Sentiment & NLP Features...
2026-04-02 15:53:28 - INFO - 🔀 Phase 5: Encoding Categorical Features...
2026-04-02 15:53:28 - INFO - 📏 Scaling Numerical Features...
2026-04-02 15:53:28 - INFO - ✅ Preprocessed data saved to: ...
2026-04-02 15:53:28 - INFO - 📊 Dataset Summary:
2026-04-02 15:53:28 - INFO - 🧪 VALIDATION: XGBoost Feature Importance Analysis
2026-04-02 15:53:29 - INFO - 📈 Model Performance: ...
2026-04-02 15:53:29 - INFO - 🔝 Top 20 Important Features: ...
2026-04-02 15:53:29 - INFO - ✅ PREPROCESSING PIPELINE COMPLETE
```

---

## Integration with ML Pipeline

### With scikit-learn
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = run_preprocessing_pipeline('input.csv')
X = df.drop(['is_returned', 'order_id', 'customer_id', 'product_id'], axis=1)
y = df['is_returned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier().fit(X_train, y_train)
```

### With neural networks (TensorFlow/PyTorch)
```python
import tensorflow as tf

df = run_preprocessing_pipeline('input.csv')
# Preprocessed data ready for keras/tf.data pipelines
X = df.drop(['is_returned', ...], axis=1).values.astype('float32')
y = df['is_returned'].values.astype('int32')

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
```

---

## Maintenance

### Version Info
- Script Version: 1.0
- Last Updated: April 2, 2026
- Compatible Python: 3.8+
- Last Tested: April 2, 2026

### Change Log
- v1.0 (Apr 2, 2026): Initial release with 5 phases

---

**Questions?** Refer to `../reports/PREPROCESSING_REPORT.md` for detailed documentation.
