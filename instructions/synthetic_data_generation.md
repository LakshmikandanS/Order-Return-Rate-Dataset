# 🎯 Synthetic Data Design (Production-Quality)

---

# 📌 Objective

Create a **highly realistic synthetic dataset** that:

* Mimics real e-commerce operations
* Preserves **causal relationships**
* Is **ML-ready without leakage**

---

# 🧠 1. Core Design Principles

## 1.1 Causality (Most Important)

Data must follow real-world logic:

* Delivery delay ↑ → Returns ↑
* Discount ↑ → Returns ↑
* Distance ↑ → Delay ↑
* Clothing → Size-related returns

❗ Avoid random labeling

---

## 1.2 Controlled Randomness

Use:

* **Probabilistic rules (not hard rules)**
* Add noise to avoid overfitting patterns

---

## 1.3 Realistic Distributions

| Feature  | Distribution           |
| -------- | ---------------------- |
| Price    | Right-skewed           |
| Quantity | Mostly 1–2             |
| Distance | Mixed (urban + remote) |
| Returns  | 10–40%                 |

---

# 🏗️ 2. Entity Design

## Customers

* 100–500 customers
* Each has:

  * base_return_tendency ∈ [0.05, 0.4]

---

## Products

* Categories:

  * Electronics
  * Clothing
  * Home
  * Beauty

* Each product has:

  * defect_rate (low but non-zero)

---

## Cities (Replace Region)

```python
cities = ["Chennai","Bangalore","Mumbai","Delhi","Hyderabad",
          "Pune","Kolkata","Ahmedabad"]
```

---

# ⚙️ 3. Data Generation Logic

## Step 1: Generate Base Variables

```python
customer_id = random.choice(customers)
product_category = random.choice(categories)
price = np.random.lognormal(mean=7, sigma=0.5)
quantity = np.random.choice([1,2,3], p=[0.6,0.3,0.1])
```

---

## Step 2: Discount Behavior

```python
discount_percentage = np.random.choice(
    [0,10,20,30,40],
    p=[0.4,0.2,0.2,0.15,0.05]
)
```

---

## Step 3: Logistics Modeling

### Distance (City Pair Logic)

```python
if customer_city == warehouse_city:
    distance_km = np.random.randint(10,200)
else:
    distance_km = np.random.randint(200,2000)
```

---

### Delivery Time

```python
expected_days = np.random.randint(2,6)

delay_prob = min(0.1 + distance_km/3000, 0.7)

delay = np.random.choice(
    [0,1,2,3,4],
    p=[1-delay_prob,0.2,0.15,0.1,delay_prob]
)

actual_days = expected_days + delay
delivery_delay = delay
```

---

## Step 4: Return Probability Model (CORE)

Construct probability instead of rules:

```python
p_return = 0

# Customer behavior
p_return += customer_base_rate

# Category effect
if product_category == "Clothing":
    p_return += 0.15

# Discount effect
p_return += discount_percentage / 200   # scaled

# Delay effect
p_return += delivery_delay * 0.08

# Distance effect
if distance_km > 1000:
    p_return += 0.05

# Noise
p_return += np.random.normal(0,0.02)

p_return = np.clip(p_return, 0, 0.9)
```

---

## Step 5: Final Return Decision

```python
is_returned = np.random.rand() < p_return
```

---

## Step 6: Return Reason Assignment

Only if returned:

```python
if not is_returned:
    return_reason = "NO_RETURN"

else:
    probs = {}

    probs["DELIVERY_DELAY"] = 0.4 if delivery_delay > 2 else 0.05
    probs["SIZE_FIT_ISSUE"] = 0.4 if product_category == "Clothing" else 0.05
    probs["QUALITY_DEFECT"] = 0.1
    probs["NO_LONGER_NEEDED"] = discount_percentage / 100
    probs["WRONG_ITEM"] = 0.05
    probs["NOT_AS_DESCRIBED"] = 0.1

    # Normalize
    total = sum(probs.values())
    probs = {k: v/total for k,v in probs.items()}

    return_reason = np.random.choice(list(probs.keys()), p=list(probs.values()))
```

---

# 📊 4. Validation Rules (MANDATORY)

## Statistical Checks

```python
# Return rate
df['is_returned'].mean()  # Expect 0.1–0.4

# Delay correlation
df.groupby('delivery_delay')['is_returned'].mean()

# Discount correlation
df.groupby('discount_percentage')['is_returned'].mean()
```

---

## Logical Checks

```python
assert all(df['order_value'] == df['product_price'] * df['quantity'])
assert all(df['delivery_delay'] >= 0)
```

---

## Behavioral Checks

* Clothing → higher returns ✔
* High delay → high returns ✔
* Remote → more delay ✔

---

# ⚠️ 5. Common Mistakes to Avoid

❌ Pure random labels
❌ Hard rules without probability
❌ Uniform distributions everywhere
❌ Unrealistic 50% return rate
❌ Using future info

---

# 🧾 Final Output Schema

```
order_id
customer_id
product_id
product_category
product_price
quantity
order_value
discount_percentage
discount_amount
customer_city
warehouse_city
distance_km
is_remote_area
shipping_mode
expected_delivery_days
actual_delivery_days
delivery_delay
is_returned
return_reason
```

---

# 🚀 Summary

This approach ensures:

* Realistic patterns
* Controlled randomness
* Strong ML learning signals
* No data leakage

👉 Result: **Industry-grade synthetic dataset suitable for modeling and evaluation**
