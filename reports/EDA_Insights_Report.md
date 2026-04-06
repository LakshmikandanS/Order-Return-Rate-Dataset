# Synthetic Data EDA & ML Metrics Report

```text
============================================================
                  AUTOMATED EDA REPORT
============================================================

-----------------------------------------
1. DATASET OVERVIEW
-----------------------------------------
Rows: 10000, Columns: 51
Columns: ['order_id', 'customer_id', 'product_id', 'order_date', 'order_day_of_week', 'order_hour', 'quantity', 'product_price', 'discount_amount', 'discount_percentage', 'final_price', 'payment_method', 'is_cod', 'shipping_mode', 'courier_partner', 'warehouse_city', 'delivery_city', 'expected_delivery_days', 'actual_delivery_days', 'delivery_delay', 'distance_km', 'is_remote_area', 'delivery_attempts', 'courier_delay_rate', 'warehouse_processing_time', 'is_returned', 'return_reason', 'return_days_after_delivery', 'city', 'state', 'pincode', 'customer_tenure_days', 'total_orders', 'total_returns', 'overall_return_rate', 'avg_order_value', 'avg_days_between_orders', 'last_order_days_ago', 'preferred_category', 'frequent_return_flag', 'product_name', 'category', 'brand', 'price', 'price_band', 'product_return_rate', 'category_return_rate', 'avg_rating', 'rating_variance', 'size_variants_count', 'is_fragile']

-----------------------------------------
2. TARGET VARIABLE DISTRIBUTION (is_returned)
-----------------------------------------
is_returned
0    66.49
1    33.51
Name: proportion, dtype: float64

-----------------------------------------
3. BIVARIATE INSIGHTS (Drivers of Returns)
-----------------------------------------
--- Average Return Rate by Delivery Delay (Days) ---
delivery_delay
(-1, 0]    25.157817
(0, 2]     38.266222
(2, 5]     66.034156
(5, 20]    82.978723

--- Average Return Rate by Payment Method ---
payment_method
COD     0.400245
Card    0.276255
UPI     0.297094

--- Average Return Rate by Category ---
category
Apparel            0.410959
Electronics        0.261042
Footwear           0.329677
Home Appliances    0.333527

-----------------------------------------
4. PREDICTIVE MODELING (MACHINE LEARNING VALUE)
-----------------------------------------
Training RandomForest to showcase predictability & feature correlation.
R-Squared (Variance Explained): 0.3207  <-- POSITIVE R-SQUARE ACHIEVED
RMSE                          : 0.3856
MAE                           : 0.3056
Classification Accuracy       : 0.7895
ROC-AUC Score                 : 0.8460

--- Top 10 Feature Importances ---
overall_return_rate     0.116165
discount_percentage     0.078657
delivery_delay          0.066460
total_returns           0.061409
distance_km             0.060486
discount_amount         0.048423
actual_delivery_days    0.040765
final_price             0.040247
avg_order_value         0.039220
courier_delay_rate      0.038505

============================================================
                      CONCLUSION
============================================================
The dataset demonstrates realistic relationships (Logistics Delay, Product Category,
Payment Method) translating directly into strong ML predictive power. Positive R-square
verifies that the synthetic rules created meaningful variance, preventing a '0' predictability scenario.
```
