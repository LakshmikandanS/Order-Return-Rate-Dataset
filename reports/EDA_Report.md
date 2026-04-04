# EDA & Validation Report — Synthetic E-Commerce Dataset

Date: 2026-04-01

## Objective
Produce and validate a production-quality synthetic e-commerce dataset that mimics real-world behavior while preserving causal relationships important for return prediction modeling.

## Dataset Summary
- Rows: 5,000 orders
- Columns: 29
- Files created: [data/synthetic_ecommerce_orders.csv](../data/synthetic_ecommerce_orders.csv), [data/customers.csv](../data/customers.csv), [data/products.csv](../data/products.csv)

## Data Integrity & Logical Checks
All logical checks passed in the validation script:
- No missing values (except missing reviews deliberately left missing).
- `order_value == product_price * quantity`: ✅
- `actual_delivery_days == expected_delivery_days + delivery_delay`: ✅
- `discount_amount` matches computed discount within tolerance: ✅
- Non-returned orders have `return_reason = NO_RETURN` and returned orders have a valid reason: ✅

## Key Summary Statistics
- Overall return rate: **36.58%** (target range: 10%–40%)
- Product price skewness: **2.599** (right-skewed as expected)
- Quantity distribution: ~60% qty=1, ~30% qty=2, ~10% qty=3
- Customer base return tendency (mean): **18.6%** (std **10.8%**)
- Product defect rate mean: **0.0322**, max **0.15**

## Causal Relationship Checks (Findings)
These checks confirm the synthetic logic follows the design principles.

1. Delivery Delay → Returns
   - Correlation (delivery_delay vs is_returned): **0.887**
   - Interpretation: Strong positive relationship; orders with higher delivery delays have markedly higher return probabilities.

2. Discount → Returns
   - Correlation (discount_percentage vs is_returned): **0.721**
   - Interpretation: Higher discounts are associated with increased returns (impulse buys / buyer remorse). The synthetic generator increased the discount effect signal to ensure this causal relationship is visible.

3. Category Effects
   - Clothing return rate: **39.60%**
   - Other categories average: **33.61%**
   - Interpretation: Clothing shows higher returns, and `SIZE_FIT_ISSUE` is the dominant reason among returned clothing orders.

4. Distance → Delivery Delay
   - Correlation (distance_km vs delivery_delay): **0.296**
   - Interpretation: As geographic distance increases, average delivery delay modestly increases; remote areas show higher average delays.

## Multicollinearity Analysis (VIF & Correlation)
Consistent with the schema-aware preprocessing guidelines, we computed the Variance Inflation Factor (VIF) on numerical features:
- **Infinite VIF ($\infty$)**: `expected_delivery_days`, `actual_delivery_days`, and `delivery_delay`. This is because `actual_delivery_days = expected_delivery_days + delivery_delay`. 
- **High VIF ($> 10$)**: `order_value` (14.7) and `product_price` (10.2). This is expected because `order_value = product_price * quantity`.
- **Moderate VIF ($5-10$)**: `quantity` (8.5).
- **Low VIF ($< 5$)**: `discount_amount`, `discount_percentage`, and `distance_km`.

**Mitigation Rule for Modeling**: Tree-based models (XGBoost, Random Forest) will naturally handle this. However, for linear models (like Logistic Regression), we must drop derived features like `actual_delivery_days`, `order_value`, and `discount_amount` to prevent singularity and unstable weights.

## Return Reasons (Normalized Frequencies among returned orders)
- QUALITY_DEFECT: 25.53%
- NOT_AS_DESCRIBED: 18.70%
- DELIVERY_DELAY: 17.93%
- NO_LONGER_NEEDED: 15.91%
- SIZE_FIT_ISSUE: 15.25%
- WRONG_ITEM: 6.67%

These proportions align with expected e-commerce patterns where quality/fit/delivery are primary causes.

## Visualizations Produced (in `EDA_Synthetic_Data.ipynb`)
- Price histogram with KDE (right-skew confirmation).
- Quantity count plot (1 vs 2 vs 3 units).
- Return distribution bar chart.
- Return rate by delivery delay (bar chart).
- Return rate by discount percentage (line chart).
- Return rate by product category (bar chart).
- Return reasons for clothing (horizontal count plot).
- Average delivery delay by distance bins and remote vs non-remote comparisons.
- Correlation Matrix heatmap across numerical features (`delivery_delay`, `order_value`, `product_price`, etc.).

Each plot visually confirms the numeric summaries above and verifies the causal relationships implemented in the generator.

## Conclusions
- The synthetic dataset successfully encodes realistic, testable causal relationships useful for building return-prediction models.
- Logical integrity checks passed; the generated dataset is ML-ready (no leakage and consistent timestamps/ordering).
- The return-rate and reason distributions are realistic and interpretable.

## Recommendations / Next Steps
- Use the dataset to prototype a classification model (e.g., XGBoost, LightGBM). Start with features: `delivery_delay`, `discount_percentage`, `product_category`, `distance_km`, `product_price`, and customer `base_return_tendency`.
- If you require lower/higher baseline return rates, adjust the `p_return` coefficients in `synthetic_data_generation.py` and re-run generation.
- Add time-based seasonality if needed (e.g., festival sale spikes) to further increase realism.

---
Report generated from the EDA performed by `EDA_Synthetic_Data.ipynb` and validation outputs from `synthetic_data_generation.py`.
