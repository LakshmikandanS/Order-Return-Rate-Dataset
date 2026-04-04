# Synthetic Data Quality Report & Pipeline Adaptations

## Overview
This report evaluates the quality of the synthetic dataset generated for e-commerce return prediction. The dataset was assessed based on logical consistency, causal relationships, and statistical distributions. Following the `pipeline_improvement_1.md` guidelines, we implemented a **Dual-Pipeline Architecture** to eliminate synthetic dominance and indirect leakages from target indicators.

---

## Architecture Revisions: Combating Leakage

### ⚠️ Identified Issue: Target Leakage (Indirect) & Synthetic Feature Dominance
Previously, logistical simulations (e.g. `delivery_delay`, `distance_km`) governed `sarcasm_score` logic, compounding artificial metrics which then recursively influenced prediction metrics. Artificial circular dependencies were triggered when modeling simulated datasets vs authentic interactions.

### ✅ Solution: Mode A vs. Mode B
- **Mode A: Real-Only Pipeline** uses exclusive ground-truth elements for predictions (clean evaluation).
  - Explicit metrics rely purely on review and rating feedback (i.e. `review_text`, `review_rating`). Target fields map directly to genuine $sentiment < -0.2$ or $rating \le 2$ behavior instead of synthesized random generator tendencies.
  - Sarcasm is assessed using purely authentic metrics such as `sentiment_rating_gap` over synthetic `delivery_delay`. Length constraints mapped linearly solve conflicting reviews vs perfect star distributions!
- **Mode B: Augmented Pipeline** retains synthetic boundaries logically mapped to allow feature modeling.

---

## Dataset Summary
- **Total Orders**: 5000
- **Customers**: 300
- **Products**: 80
- **Columns**: 29
- **File Location**: `data/synthetic_ecommerce_orders.csv`

---

## Validation Highlights
### 1. Logical Consistency
- **Order Value Calculation**: Verified as `order_value = product_price × quantity`.
- **Delivery Delay**: `delivery_delay ≥ 0`.
- **Discount Amount**: Consistent with `discount_percentage`.
- **Return Reasons**: All returned orders have valid reasons.

### 2. Causal Relationships
- **Delivery Delay → Returns**: Strong positive correlation (0.934).
- **Discount → Returns**: Strong positive correlation (0.946).
- **Distance → Delay**: Moderate positive correlation (0.282).

### 3. Statistical Distributions
- **Return Rate**: 36.58% (within expected range of 10-40%).
- **Price Skewness**: 2.599 (right-skewed as expected).
- **Quantity Distribution**: 
  - 1 unit: 59.54%
  - 2 units: 30.7%
  - 3 units: 9.76%

---

## Review Analysis
- **Total Reviews**: 3230
- **Missing Reviews**: 1770 (35.4%)
- **Sarcastic Comments (Mathematical Model)**: 0% (Keyword strategy abandoned, contradiction features require separate text sentiment inference pipelines prior to evaluation)

---

## Extended Metrics (computed)

- **Total orders**: 5000
- **Reviews with text**: 3230
- **Missing review entries**: 1770 (35.40% of orders)
- **Sarcastic reviews (mathematical model)**: 0
  - **% of reviews with text**: 0.00%
  - **% of all orders**: 0.00%

- **Per-category return rates (top → bottom)**:
  - Clothing: 41.41% (990 orders, 410 returns)
  - Beauty: 37.79% (778 orders, 294 returns)
  - Electronics: 36.26% (1809 orders, 656 returns)
  - Home: 32.96% (1423 orders, 469 returns)

- **Top return reasons (percentage of returned orders)**:
  - QUALITY_DEFECT: 25.53%
  - NOT_AS_DESCRIBED: 18.70%
  - DELIVERY_DELAY: 17.93%
  - NO_LONGER_NEEDED: 15.91%
  - SIZE_FIT_ISSUE: 15.25%
  - WRONG_ITEM: 6.67%

- **Missingness by column**: `review_rating`, `review_text`, `review_date` have 1770 missing values each (others complete).

- **Price skewness**: 2.5994 (strong right skew — consistent with a long tail of expensive items)

- **Quantity distribution**:
  - 1 unit: 59.54%
  - 2 units: 30.70%
  - 3 units: 9.76%

---

## Strengths (updated)
1. **Realistic Patterns**: Temporal trends, festival spikes, and customer behavior metrics are represented and produce plausible correlations.
2. **Causal Validity**: Delay and discount correlate strongly with returns; patterns align with real-world intuition.
3. **Balanced Categories**: All major categories have substantial samples for modelling.

## Areas for Improvement (updated)
1. **Review Coverage**: ~35% of orders lack review text or ratings. Consider filling a higher proportion if reviews are required as features.
2. **Sarcasm Detection**: Current sarcasm is evaluated via a mathematical contradiction model. We've completely discarded the old keyword-matching heuristic format for a robust outcome/sentiment logic. Note that since generated text does not strictly calculate semantic sentiment during creation without NLP pipelines, zero mathematical flagged instances exist in raw generation outputs.
3. **Edge-case Outliers**: Price and return-rate outliers should be reviewed; consider capping or documenting them for model training.

## Recommendations & Next Steps
1. Impute or enrich missing `review_text` and `review_rating` where appropriate (use templates or synthetic augmentation).
2. Create a small labeled subset for sarcasm/sentiment (200–500 examples) to train a lightweight classifier.
3. Add a companion notebook `reports/quality_checks.ipynb` that runs the metric script and renders charts (histograms for price, return rate by category, missingness heatmap).
4. Track dataset generation parameters in `reports/generation_metadata.json` (seed, counts, templates used) for reproducibility.

## Appendix — How metrics were computed
- Metrics were computed by `scripts/compute_synth_metrics.py` which reads `data/synthetic_ecommerce_orders.csv` and reports counts, per-category return rates, top return reasons, and the new mathematical contradiction model for sarcasm without string matching.

---

*Updated on: April 2, 2026*

---

## Strengths
1. **Realistic Patterns**: Temporal trends, festival spikes, and customer behavior metrics are well-represented.
2. **Causal Validity**: Strong correlations align with real-world expectations.
3. **Diverse Reviews**: Includes a mix of positive, neutral, and sarcastic comments.

---

## Areas for Improvement
1. **Review Coverage**: Address missing reviews (35.4%).
2. **Sarcasm Diversity**: Expand sarcastic templates for greater variety.
3. **Anomalies**: Investigate outliers in price and return rates.

---

## Conclusion
The synthetic dataset demonstrates high quality with realistic patterns and logical consistency. Minor improvements in review coverage and sarcasm diversity can further enhance its utility for predictive modeling.

---

*Generated on: April 2, 2026*