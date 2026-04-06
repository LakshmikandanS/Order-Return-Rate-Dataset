import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
NUM_CUSTOMERS = 1000
NUM_ORDERS = 10000

# -----------------------------
# CITIES
# -----------------------------
cities = [
    ("Bangalore", "Karnataka"), ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"),
    ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"),
    ("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh")
]

courier_partners = ["Delhivery", "BlueDart", "Ecom Express"]

# -----------------------------
# PRODUCTS
# -----------------------------
products = [
    ("P1", "Nike Air Zoom Pegasus 40", "Footwear", "Nike", 11999),
    ("P2", "Levi's Slim Fit T-Shirt", "Apparel", "Levi's", 1799),
    ("P3", "Samsung Galaxy S21 FE", "Electronics", "Samsung", 39999),
    ("P4", "Sony WH-1000XM4 Headphones", "Electronics", "Sony", 24990),
    ("P5", "Levi's 511 Slim Jeans", "Apparel", "Levi's", 3499),
    ("P6", "Philips HL7756 Mixer Grinder", "Home Appliances", "Philips", 3199)
]

product_df = pd.DataFrame(products, columns=[
    "product_id", "product_name", "category", "brand", "price"
])

product_df["price_band"] = pd.cut(product_df["price"], bins=[0, 3000, 15000, 50000], labels=["low", "mid", "high"])
# Adding 10-15% variance to product return tendencies
product_df["product_return_rate"] = [0.12, 0.25, 0.08, 0.05, 0.22, 0.07]
product_df["category_return_rate"] = product_df.groupby("category")["product_return_rate"].transform("mean")
product_df["avg_rating"] = [4.5, 4.0, 4.3, 4.8, 3.9, 4.2]
product_df["rating_variance"] = [0.4, 0.8, 0.3, 0.2, 0.9, 0.5]
product_df["size_variants_count"] = [8, 5, 1, 1, 6, 1]
product_df["is_fragile"] = [0, 0, 1, 1, 0, 1]

product_df.to_csv("data/products.csv", index=False)

# -----------------------------
# CUSTOMERS (LEAKAGE REMOVED VIA HISTORICAL ESTIMATION)
# -----------------------------
customers = []
segments = (
    ["zero"] * int(NUM_CUSTOMERS * 0.2) +
    ["low"] * int(NUM_CUSTOMERS * 0.4) +
    ["medium"] * int(NUM_CUSTOMERS * 0.25) +
    ["high"] * int(NUM_CUSTOMERS * 0.15)
)
random.shuffle(segments)

# Dictionary to hold the true latent rates so we can use them later
customer_latent_rates = {}

for i in range(NUM_CUSTOMERS):
    seg = segments[i]
    if seg == "zero": return_rate = 0.02
    elif seg == "low": return_rate = random.uniform(0.04, 0.1)
    elif seg == "medium": return_rate = random.uniform(0.15, 0.22)
    else: return_rate = random.uniform(0.25, 0.4)
    
    # Store true latent rate (hidden from ML model)
    customer_latent_rates[f"C{i+1}"] = return_rate

    city, state = random.choice(cities)
    
    # Generate completely independent 'historical' values representing the customer BEFORE this dataset
    hist_orders = random.randint(5, 50)
    # Give it noise so the history doesn't perfectly map to the true latent probability
    hist_returns = np.random.binomial(hist_orders, np.clip(return_rate + np.random.normal(0, 0.05), 0.01, 0.99))
    hist_overall_rate = hist_returns / hist_orders
    
    customers.append([
        f"C{i+1}", city, state, random.randint(100000, 999999), random.randint(30, 900),
        hist_orders, hist_returns, round(hist_overall_rate, 2), random.randint(500, 20000), random.uniform(5, 30),
        random.randint(1, 60), random.choice(["Apparel", "Electronics", "Footwear", "Home Appliances"]),
        1 if hist_overall_rate > 0.3 else 0
    ])

customer_df = pd.DataFrame(customers, columns=[
    "customer_id", "city", "state", "pincode", "customer_tenure_days",
    "total_orders", "total_returns", "overall_return_rate",
    "avg_order_value", "avg_days_between_orders",
    "last_order_days_ago", "preferred_category", "frequent_return_flag"
])

# -----------------------------
# ORDERS + LOGISTICS + RETURNS
# -----------------------------
orders, logistics, returns = [], [], []
start_date = datetime(2025, 1, 1)

def get_shipping_days(mode):
    if mode == "Same-Day": return 0, 1
    elif mode == "Express": return 1, 3
    else: return 3, 6

for i in range(NUM_ORDERS):
    order_id = f"O{i+1}"
    cust = customer_df.sample(1).iloc[0]
    prod = product_df.sample(1).iloc[0]

    # TEMPORAL DRIFT: Add seasonal spikes based on standard months (e.g., Nov/Dec = Holiday, higher impulse buys)
    days_offset = random.randint(0, 365)
    order_date = start_date + timedelta(days=days_offset)
    month = order_date.month
    
    quantity = np.random.choice([1, 2, 3], p=[0.85, 0.1, 0.05])
    # Add random perturbations to discount percentage
    discount_pct = random.choice([0, 10, 20, 30, 40]) + np.random.randint(-2, 3) if random.random() > 0.5 else random.choice([0, 10, 20, 30])
    discount_pct = np.clip(discount_pct, 0, 80)
    discount_amt = prod["price"] * (discount_pct / 100) * quantity
    final_price = (prod["price"] * quantity) - discount_amt
    payment_method = np.random.choice(["COD", "UPI", "Card"], p=[0.4, 0.4, 0.2])

    # Logistics
    shipping_mode = np.random.choice(["Standard", "Express", "Same-Day"], p=[0.6, 0.3, 0.1])
    expected_min, expected_max = get_shipping_days(shipping_mode)
    expected_days = random.randint(expected_min, expected_max)

    distance = random.randint(5, 2000)
    is_remote = 1 if distance > 800 else 0

    # Smooth delay logic
    delay_prob = 0.1 + (0.15 * (distance/2000)) + (0.1 if shipping_mode == "Standard" else 0)
    delay = np.random.poisson(delay_prob * 2.5) 
    
    # Perturb delay to add imperfection
    if random.random() < 0.05: delay = random.randint(3, 7) # Occasional random huge delays
    actual_days = expected_days + delay

    # -----------------------------
    # LOGIT MODEL - SOFT REALISTIC PROBABILITIES
    # -----------------------------
    latent_rate = customer_latent_rates[cust["customer_id"]]
    logit_p = np.log(max(latent_rate, 0.001) / (1 - min(latent_rate, 0.999))) * 2.2 - 0.7
    
    # Strong Continuous Effects for ML predictability
    logit_p += 0.07 * discount_pct          
    logit_p += 0.85 * delay                  
    logit_p -= 2.0 * (prod["avg_rating"] - 4.0) 
    
    if payment_method == "COD": logit_p += 1.4
    if prod["category"] in ["Apparel", "Footwear"]: logit_p += 1.2
    
    # Hidden Interactions
    if discount_pct > 25 and prod["category"] in ["Apparel", "Footwear"]:
        logit_p += 2.0
    if payment_method == "COD" and delay > 2:
        logit_p += 1.8
    if prod["is_fragile"] and (distance > 1000):
        logit_p += 1.5 * (delay + 1)
        
    # Temporal Drift & Seasonality
    if month in [1, 10, 11, 12]:
        logit_p += 0.7
    logit_p += 0.06 * (month)

    # Gaussian Noise tightly controlled
    logit_p += np.random.normal(0, 0.03)
    
    prob = 1 / (1 + np.exp(-logit_p))
    
    # Ensure no exact 0 or 1 boundaries but highly polarized
    prob = np.clip(prob, 0.01, 0.96)
    
    # Sample actual outcome
    is_returned = 1 if random.random() < prob else 0

    # Inject 3% Random Anomalies (Complete contradiction)
    if random.random() < 0.03:
        is_returned = 1 - is_returned

    # -----------------------------
    # GENERATE REASONS
    # -----------------------------
    reason = "NONE"
    if is_returned:
        options = ["NO_LONGER_NEEDED", "WRONG_ITEM", "CHANGED_MIND"]
        if delay > 2 and random.random() > 0.3: options.append("DELIVERY_DELAY")
        if quantity > 1 and prod["size_variants_count"] > 1: options.append("SIZE_FIT_ISSUE")
        if prod["category"] in ["Apparel", "Footwear"]: options.extend(["SIZE_FIT_ISSUE"] * 3)
        if prod["category"] == "Electronics": options.extend(["QUALITY_DEFECT"] * 2)
        if prod["is_fragile"] and distance > 800: options.extend(["DAMAGED_IN_TRANSIT"] * 2)
        if prod["avg_rating"] < 4.0: options.extend(["NOT_AS_DESCRIBED"] * 2)
        reason = random.choice(options)

    orders.append([
        order_id, cust["customer_id"], prod["product_id"], order_date.date(),
        order_date.strftime("%A"), random.randint(0, 23), quantity, prod["price"],
        round(discount_amt, 2), discount_pct, round(final_price, 2), payment_method, 1 if payment_method == "COD" else 0
    ])

    logistics.append([
        order_id, shipping_mode, random.choice(courier_partners), random.choice(cities)[0], cust["city"],
        expected_days, actual_days, delay, distance, is_remote, random.randint(1, 3),
        round(random.uniform(0.05, 0.3) + (0.1 if delay > 0 else 0), 2), round(random.uniform(1, 3), 2)
    ])

    returns.append([
        order_id, is_returned, reason, random.randint(1, 10) if is_returned else 0
    ])

orders_df = pd.DataFrame(orders, columns=[
    "order_id", "customer_id", "product_id", "order_date", "order_day_of_week", "order_hour",
    "quantity", "product_price", "discount_amount", "discount_percentage", "final_price", "payment_method", "is_cod"
])

logistics_df = pd.DataFrame(logistics, columns=[
    "order_id", "shipping_mode", "courier_partner", "warehouse_city", "delivery_city", "expected_delivery_days",
    "actual_delivery_days", "delivery_delay", "distance_km", "is_remote_area", "delivery_attempts",
    "courier_delay_rate", "warehouse_processing_time"
])

returns_df = pd.DataFrame(returns, columns=[
    "order_id", "is_returned", "return_reason", "return_days_after_delivery"
])

# NOTE: We DO NOT recalculate the target leak features (`total_returns`, `overall_return_rate`). 
# They remain as historical aggregates from before this dataset window. This is proper ML leakage prevention.

# -----------------------------
# SAVE FILES
# -----------------------------
customer_df.to_csv("data/customers.csv", index=False)
orders_df.to_csv("data/orders.csv", index=False)
logistics_df.to_csv("data/logistics.csv", index=False)
returns_df.to_csv("data/returns.csv", index=False)
merged_df = orders_df.merge(logistics_df, on="order_id").merge(returns_df, on="order_id").merge(customer_df, on="customer_id").merge(product_df, on="product_id")
merged_df.to_csv("data/final_combined_data.csv", index=False)

print("✅ Data generation complete - Fully realized real-world variance!")
print("✅ Leakages mitigated. Noise and hidden interactions successfully injected.")
