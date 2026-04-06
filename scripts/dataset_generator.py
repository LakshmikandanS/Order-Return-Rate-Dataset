import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
NUM_CUSTOMERS = 500
NUM_ORDERS = 10000

# Ensure directories exist
os.makedirs("data", exist_ok=True)

# -----------------------------
# CITIES
# -----------------------------
cities = [
    ("Bangalore", "Karnataka"), ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"),
    ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"),
    ("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh"), ("Pune", "Maharashtra"),
    ("Ahmedabad", "Gujarat"), ("Gurgaon", "Haryana"), ("Noida", "Uttar Pradesh")
]

# -----------------------------
# PRODUCTS
# -----------------------------
# We use the schema from dataset_generator (1).py
product_data = [
    ("P1", "Nike Air Zoom Pegasus 40", "Footwear", "Nike", 11999),
    ("P2", "Levi's Slim Fit T-Shirt", "Apparel", "Levi's", 1799),
    ("P3", "Samsung Galaxy S21 FE", "Electronics", "Samsung", 39999),
    ("P4", "Sony WH-1000XM4 Headphones", "Electronics", "Sony", 24990),
    ("P5", "Levi's 511 Slim Jeans", "Apparel", "Levi's", 3499),
    ("P6", "Philips HL7756 Mixer Grinder", "Home Appliances", "Philips", 3199),
    ("P7", "Adidas Ultraboost 22", "Footwear", "Adidas", 17999),
    ("P8", "Apple iPhone 14", "Electronics", "Apple", 69900),
    ("P9", "U.S. Polo Assn. Casual Shirt", "Apparel", "U.S. Polo", 2499),
    ("P10", "Dell XPS 13 Laptop", "Electronics", "Dell", 95000)
]

product_df = pd.DataFrame(product_data, columns=[
    "product_id", "product_name", "category", "brand", "price"
])

# Adding attributes that drive returns
# High price + high return rate (size/fit issues in apparel)
product_df["product_return_rate"] = [0.12, 0.25, 0.08, 0.05, 0.22, 0.07, 0.15, 0.04, 0.20, 0.06]
product_df["avg_rating"] = [4.5, 4.0, 4.3, 4.8, 3.9, 4.2, 4.6, 4.9, 4.1, 4.7]
product_df["is_fragile"] = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]

# -----------------------------
# CUSTOMERS
# -----------------------------
customers = []
for i in range(NUM_CUSTOMERS):
    cust_id = f"C{i+1}"
    city, state = random.choice(cities)
    tenure = random.randint(30, 1200)
    
    # Customer baseline return behavior (Latent variable)
    # This creates a "Bad Actor" vs "Loyal Customer" distribution
    behavior_type = np.random.choice(['Loyal', 'Regular', 'Frequent Returner'], p=[0.2, 0.6, 0.2])
    if behavior_type == 'Loyal':
        base_rate = random.uniform(0.01, 0.05)
    elif behavior_type == 'Regular':
        base_rate = random.uniform(0.06, 0.15)
    else:
        base_rate = random.uniform(0.20, 0.45)

    customers.append([
        cust_id, city, state, random.randint(110001, 800000), tenure,
        base_rate, random.choice(["Apparel", "Electronics", "Footwear", "Home Appliances"])
    ])

customer_df = pd.DataFrame(customers, columns=[
    "customer_id", "city", "state", "pincode", "customer_tenure_days",
    "internal_base_rate", "preferred_category"
])

# -----------------------------
# ORDERS + LOGISTICS + RETURNS
# -----------------------------
orders = []
logistics = []
returns = []

start_date = datetime(2025, 1, 1)

for i in range(NUM_ORDERS):
    order_id = f"ORD{i+1}"
    cust = customer_df.sample(1).iloc[0]
    prod = product_df.sample(1).iloc[0]

    order_date = start_date + timedelta(days=random.randint(0, 365))
    
    # Quantity weight: Higher quantity slightly increases return risk (ordering multiple sizes)
    quantity = np.random.choice([1, 2, 3], p=[0.85, 0.1, 0.05])
    
    # Discount impact: Heavy discounts attract bargain hunters who might return more often
    discount_pct = np.random.choice([0, 10, 20, 50], p=[0.4, 0.3, 0.2, 0.1])
    discount_amt = prod["price"] * (discount_pct / 100)
    final_price = (prod["price"] - discount_amt) * quantity

    payment_method = random.choice(["COD", "UPI", "Card"])
    is_cod = 1 if payment_method == "COD" else 0

    # Logistics - Impact on Return
    shipping_mode = random.choice(["Standard", "Express"])
    expected_days = 2 if shipping_mode == "Express" else 5
    
    # Adding a correlation: Longer distances = higher chance of delay
    distance = random.randint(50, 2500)
    is_remote = 1 if distance > 1500 else 0
    
    # Delay generation (The key driver for R-square in logistics)
    delay_prob = 0.1 + (0.2 if is_remote else 0) + (0.1 if shipping_mode == "Standard" else 0)
    delay = np.random.poisson(delay_prob * 2) 
    actual_days = expected_days + delay

    # -----------------------------
    # CALCULATE RETURN PROBABILITY (Real-world Modeling)
    # -----------------------------
    # Logit-like approach to ensure features have strong predictive power
    # 1. Customer baseline
    logit_p = np.log(cust["internal_base_rate"] / (1 - cust["internal_base_rate"]))
    
    # 2. Category penalty (Apparel/Footwear have higher returns due to fit)
    if prod["category"] in ["Apparel", "Footwear"]:
        logit_p += 1.2
        
    # 3. Delay penalty (Logistic regression feature)
    if delay > 0:
        logit_p += 0.5 * delay
        
    # 4. COD penalty (Cash on delivery has higher return rates in India)
    if is_cod:
        logit_p += 0.8
        
    # 5. Rating boost (Good products are returned less)
    logit_p -= 0.6 * (prod["avg_rating"] - 3.5)
    
    # 6. Fragile + Remote = Damage risk
    if prod["is_fragile"] and is_remote:
        logit_p += 1.5

    # Convert Logit to Probability
    prob = 1 / (1 + np.exp(-logit_p))
    
    # Adding subtle noise so R-square isn't 1.0 (Realism)
    prob = np.clip(prob + np.random.normal(0, 0.05), 0, 1)

    is_returned = 1 if random.random() < prob else 0

    # Return reason logic
    reason = "NONE"
    if is_returned:
        if delay > 2: reason = "DELIVERY_DELAY"
        elif prod["category"] in ["Apparel", "Footwear"] and random.random() > 0.5: reason = "SIZE_FIT_ISSUE"
        elif prod["is_fragile"] and is_remote and random.random() > 0.4: reason = "DAMAGED_IN_TRANSIT"
        elif prod["avg_rating"] < 4.0: reason = "NOT_AS_DESCRIBED"
        else: reason = "NO_LONGER_NEEDED"

    # Save data
    orders.append([
        order_id, cust["customer_id"], prod["product_id"],
        order_date.date(), quantity, prod["price"], discount_amt,
        discount_pct, final_price, payment_method, is_cod
    ])

    logistics.append([
        order_id, shipping_mode, expected_days, actual_days,
        delay, distance, is_remote
    ])

    returns.append([
        order_id, is_returned, reason,
        random.randint(1, 7) if is_returned else 0
    ])

# Create DataFrames
orders_df = pd.DataFrame(orders, columns=[
    "order_id", "customer_id", "product_id", "order_date",
    "quantity", "product_price", "discount_amount", "discount_percentage",
    "final_price", "payment_method", "is_cod"
])

logistics_df = pd.DataFrame(logistics, columns=[
    "order_id", "shipping_mode", "expected_delivery_days", "actual_delivery_days",
    "delivery_delay", "distance_km", "is_remote_area"
])

returns_df = pd.DataFrame(returns, columns=[
    "order_id", "is_returned", "return_reason", "return_days_after_delivery"
])

# -----------------------------
# FINAL CUSTOMER AGGREGATION
# -----------------------------
# Update customer_df with real calculated stats for the CSV
cust_order_stats = orders_df.groupby("customer_id")["order_id"].count()
cust_return_stats = returns_df.merge(orders_df, on="order_id").groupby("customer_id")["is_returned"].sum()

customer_df["total_orders"] = customer_df["customer_id"].map(cust_order_stats).fillna(0)
customer_df["total_returns"] = customer_df["customer_id"].map(cust_return_stats).fillna(0)
customer_df["overall_return_rate"] = (customer_df["total_returns"] / customer_df["total_orders"]).fillna(0).round(2)
customer_df["frequent_return_flag"] = (customer_df["overall_return_rate"] > 0.3).astype(int)

# Drop internal hidden column
customer_df_final = customer_df.drop(columns=["internal_base_rate"])

# -----------------------------
# SAVE
# -----------------------------
customer_df_final.to_csv("data/customers.csv", index=False)
product_df.to_csv("data/products.csv", index=False)
orders_df.to_csv("data/orders.csv", index=False)
logistics_df.to_csv("data/logistics.csv", index=False)
returns_df.to_csv("data/returns.csv", index=False)

# Final Merged Dataset for easy EDA
merged_df = orders_df.merge(logistics_df, on="order_id").merge(returns_df, on="order_id").merge(customer_df_final, on="customer_id").merge(product_df, on="product_id")
merged_df.to_csv("data/final_combined_data.csv", index=False)

print(f"✅ Generated {NUM_ORDERS} orders with realistic correlations.")
print(f"✅ R-Square Support: Strong dependence on 'delivery_delay', 'is_cod', 'category', and 'customer_behavior'.")
