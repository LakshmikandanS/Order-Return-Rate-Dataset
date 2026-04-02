# Synthetic Data Generation Improvements

## 🎯 Objective
Make synthetic logistics data more **realistic, correlated, and ML-ready**.

---

## 1. Introduce Real-World Correlations
Avoid independent random values. Add relationships:
- Higher `delivery_delay` → lower `review_rating`
- Higher `defect_rate` → higher `is_returned`
- Longer `distance_km` → higher delay probability
- Higher `discount_percentage` → slightly higher return probability

---

## 2. Use Rule-Based Probabilities (Not Pure Randomness)
Replace randomness with controlled logic:
- Example:
  - If `delay > 3` → return probability ↑
  - If `rating ≤ 2` → return probability ↑
- Combine rules with randomness for realism

---

## 3. Maintain Logical Consistency
Ensure data integrity:
- `delivery_delay = actual_delivery_days - expected_delivery_days`
- `actual_delivery_days ≥ expected_delivery_days` (mostly)
- High `distance_km` → likely higher delays
- `is_remote_area = 1` for far distances

---

## 4. Improve Missing Data Patterns
Avoid random nulls:
- Missing `review_text` only if no review given
- Link missing fields logically (not independently)

---

## 5. Add Time-Based Behavior
Include temporal patterns:
- Weekends → more delays
- Festivals → more orders + returns
- Seasonal variations in demand

---

## 6. Model Customer Behavior
Add features:
- `past_return_rate`
- `total_orders`
- `avg_order_value`

These improve personalization and prediction accuracy.

---

## 7. Model Product Behavior
- Same product → consistent `defect_rate`
- Certain categories (e.g., Clothing) → higher return probability

---

## 8. Balance Target Variable
Ensure:
- Reasonable ratio of `is_returned` vs `not returned`
- Avoid extreme imbalance

---

## 9. Prevent Data Leakage
Exclude during training (for return prediction):
- `review_text`
- `review_rating`
- `actual_delivery_days`
- `return_reason`

---

## 10. Add Controlled Noise
- Introduce slight inconsistencies
- Avoid perfect patterns (real data is imperfect)

---

## ✅ Final Checklist
- Correlations exist
- Logical rules applied
- Time patterns included
- Customer & product behavior modeled
- No leakage features
- Realistic missing values

---

## 🧠 Key Insight
Synthetic data should be **rule-driven + slightly noisy**, not purely random.