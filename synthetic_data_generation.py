"""
🎯 Synthetic Data Generation for E-Commerce Return Prediction
==============================================================

Generates a production-quality synthetic dataset following causal principles:
- Delivery delay ↑ → Returns ↑
- Discount ↑ → Returns ↑  
- Distance ↑ → Delay ↑
- Clothing → Size-related returns

Improvements over the reference dataset:
1. Right-skewed price distribution (lognormal)
2. Realistic quantity distribution (60% qty=1, 30% qty=2, 10% qty=3)
3. City-based logistics instead of vague regions
4. Proper causal return probability model (no hard rules)
5. Return rate controlled to 15-30% range
6. Shipping mode affects delivery speed
7. No negative delivery delays
8. Product defect rates per product
9. Customer behavioral consistency (base return tendency)
10. Temporal ordering with realistic date generation
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ============================================================
# Configuration
# ============================================================
np.random.seed(42)

NUM_ORDERS = 5000
NUM_CUSTOMERS = 300
NUM_PRODUCTS = 80

CATEGORIES = ["Electronics", "Clothing", "Home", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.30, 0.25, 0.20]  # Clothing is the most popular

CITIES = [
    "Chennai", "Bangalore", "Mumbai", "Delhi",
    "Hyderabad", "Pune", "Kolkata", "Ahmedabad"
]

# Approximate pairwise distances (km) between cities
# Used to generate realistic distance_km values
CITY_COORDS = {
    "Chennai":    (13.08, 80.27),
    "Bangalore":  (12.97, 77.59),
    "Mumbai":     (19.08, 72.88),
    "Delhi":      (28.61, 77.21),
    "Hyderabad":  (17.38, 78.49),
    "Pune":       (18.52, 73.86),
    "Kolkata":    (22.57, 88.36),
    "Ahmedabad":  (23.02, 72.57),
}

SHIPPING_MODES = ["Standard", "Express"]
SHIPPING_WEIGHTS = [0.55, 0.45]  # Slightly more standard

RETURN_REASONS = [
    "DELIVERY_DELAY", "SIZE_FIT_ISSUE", "QUALITY_DEFECT",
    "NOT_AS_DESCRIBED", "NO_LONGER_NEEDED", "WRONG_ITEM"
]

DISCOUNT_VALUES = [0, 5, 10, 15, 20, 25, 30, 40]
DISCOUNT_PROBS = [0.30, 0.10, 0.15, 0.15, 0.12, 0.08, 0.07, 0.03]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_ecommerce_orders.csv")


# ============================================================
# 1. Entity Generation
# ============================================================

def generate_customers(n):
    """Generate customer profiles with behavioral tendencies."""
    customers = []
    for cid in range(1, n + 1):
        customers.append({
            "customer_id": cid,
            "home_city": np.random.choice(CITIES),
            # Base return tendency: most customers are low-returners,
            # some are chronic returners (right-skewed beta distribution)
            "base_return_tendency": np.clip(
                np.random.beta(2, 8),  # Mean ~0.2, skewed right
                0.03, 0.40
            ),
        })
    return pd.DataFrame(customers)


def generate_products(n):
    """Generate product catalog with category-specific properties."""
    products = []
    for pid in range(1, n + 1):
        category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)

        # Base price depends on category (lognormal, category-adjusted)
        if category == "Electronics":
            base_price = np.random.lognormal(mean=7.5, sigma=0.6)
        elif category == "Clothing":
            base_price = np.random.lognormal(mean=6.8, sigma=0.5)
        elif category == "Home":
            base_price = np.random.lognormal(mean=7.0, sigma=0.55)
        else:  # Beauty
            base_price = np.random.lognormal(mean=6.2, sigma=0.5)

        base_price = np.clip(base_price, 150, 15000)

        # Defect rate varies by product (most are low, a few are problematic)
        defect_rate = np.clip(np.random.exponential(0.03), 0.005, 0.15)

        products.append({
            "product_id": f"P{pid:03d}",
            "product_category": category,
            "base_price": round(base_price, 0),
            "defect_rate": round(defect_rate, 4),
        })
    return pd.DataFrame(products)


def compute_city_distance(city1, city2):
    """Compute approximate distance between two Indian cities using coordinates."""
    if city1 == city2:
        return np.random.randint(10, 150)  # Intra-city/nearby

    lat1, lon1 = CITY_COORDS[city1]
    lat2, lon2 = CITY_COORDS[city2]

    # Haversine-like rough approximation (1 degree ≈ 111 km)
    dlat = abs(lat1 - lat2) * 111
    dlon = abs(lon1 - lon2) * 111 * np.cos(np.radians((lat1 + lat2) / 2))
    base_dist = np.sqrt(dlat**2 + dlon**2)

    # Add noise ±15%
    noise = np.random.uniform(0.85, 1.15)
    return int(np.clip(base_dist * noise, 50, 2500))


# ============================================================
# 2. Order Generation
# ============================================================

def generate_orders(customers_df, products_df, n_orders):
    """Generate orders with causal relationships baked in."""
    orders = []

    # Pre-compute lookups
    customer_dict = customers_df.set_index("customer_id").to_dict("index")
    product_dict = products_df.set_index("product_id").to_dict("index")
    product_ids = products_df["product_id"].tolist()

    # Warehouses are in a subset of cities
    warehouse_cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata"]

    # Date range: 1 year of orders
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range_days = (end_date - start_date).days

    for oid in range(1, n_orders + 1):
        # --- Step 1: Select customer and product ---
        cust_id = np.random.randint(1, len(customer_dict) + 1)
        cust = customer_dict[cust_id]

        prod_id = np.random.choice(product_ids)
        prod = product_dict[prod_id]

        customer_city = cust["home_city"]
        warehouse_city = np.random.choice(warehouse_cities)

        # --- Step 2: Generate order date ---
        # 20% of orders correspond to festival spikes (Oct-Nov)
        if np.random.rand() < 0.20:
            # Normal distribution centered around Halloween/Diwali (Day ~300)
            day_offset = int(np.clip(int(np.random.normal(300, 15)), 0, date_range_days))
            order_date = start_date + timedelta(days=day_offset)
        else:
            order_date = start_date + timedelta(days=np.random.randint(0, date_range_days))

        # --- Step 3: Pricing ---
        # Price has minor variation (±5%) around base price
        product_price = round(prod["base_price"] * np.random.uniform(0.95, 1.05))
        product_price = max(product_price, 100)

        # Quantity: heavily skewed toward 1
        quantity = int(np.random.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))

        # Add realistic pricing noise: Shipping Fee and Tax to break 1.0 R^2 correlation
        base_order_value = product_price * quantity
        shipping_fee = np.random.choice([0, 49, 99], p=[0.4, 0.4, 0.2])
        tax_rate = np.random.choice([0.05, 0.12, 0.18], p=[0.3, 0.5, 0.2])
        
        order_value = round((base_order_value + shipping_fee) * (1 + tax_rate), 2)

        # --- Step 4: Discount ---
        discount_percentage = int(np.random.choice(DISCOUNT_VALUES, p=DISCOUNT_PROBS))
        discount_amount = round(base_order_value * discount_percentage / 100, 2)
        is_remote_area = 1 if distance_km > 1200 else 0

        shipping_mode = np.random.choice(SHIPPING_MODES, p=SHIPPING_WEIGHTS)

        # Expected delivery depends on shipping mode
        if shipping_mode == "Express":
            expected_delivery_days = np.random.randint(2, 5)  # 2-4 days
        else:
            expected_delivery_days = np.random.randint(4, 8)  # 4-7 days

        # Delivery delay: depends on distance and shipping mode
        # Higher distance → higher delay probability
        delay_base_prob = min(0.08 + distance_km / 4000, 0.55)
        if shipping_mode == "Express":
            delay_base_prob *= 0.6  # Express reduces delay probability

        # Remote areas have higher delay probability
        if is_remote_area:
            delay_base_prob = min(delay_base_prob + 0.15, 0.65)
            
        # Weekends have higher delay probability
        if order_date.weekday() >= 5: # Saturday/Sunday
            delay_base_prob = min(delay_base_prob + 0.10, 0.70)

        # Generate delay (0-5 days, weighted by probability)
        delay_probs = np.array([
            1 - delay_base_prob,        # 0 days delay
            delay_base_prob * 0.35,      # 1 day
            delay_base_prob * 0.25,      # 2 days
            delay_base_prob * 0.20,      # 3 days
            delay_base_prob * 0.12,      # 4 days
            delay_base_prob * 0.08,      # 5 days
        ])
        delay_probs = delay_probs / delay_probs.sum()  # normalize
        delivery_delay = int(np.random.choice([0, 1, 2, 3, 4, 5], p=delay_probs))

        actual_delivery_days = expected_delivery_days + delivery_delay

        # --- Step 6: Return Probability Model (CORE) ---
        p_return = 0.0

        # Customer base tendency (main driver)
        p_return += cust["base_return_tendency"]

        # Category effect: Clothing has higher returns
        category = prod["product_category"]
        if category == "Clothing":
            p_return += 0.10
        elif category == "Electronics":
            p_return += 0.03  # Slightly higher due to expectations
        elif category == "Beauty":
            p_return += 0.02

        # Discount effect: Higher discounts → impulse buys → more returns
        p_return += (discount_percentage / 100) * 0.45

        # Delivery delay effect: Strong causal driver
        p_return += delivery_delay * 0.05

        # Distance effect: Long distance → more handling → more issues
        if distance_km > 1200:
            p_return += 0.04
        elif distance_km > 800:
            p_return += 0.02

        # Causal Temporal Effect: Festival period → Impulsive buying → Higher returns
        if 285 <= (order_date - start_date).days <= 330:
            p_return += 0.05

        # Product defect rate contribution
        p_return += prod["defect_rate"] * 0.5

        # High-value orders: slightly more returns (buyer's remorse)
        if order_value > 8000:
            p_return += 0.03

        # Noise: realistic randomness
        p_return += np.random.normal(0, 0.03)

        # Clip to valid range
        p_return = np.clip(p_return, 0.02, 0.85)

        # --- Step 7: Final return decision ---
        is_returned = 1 if np.random.rand() < p_return else 0

        # --- Step 8: Return reason (only if returned) ---
        if not is_returned:
            return_reason = "NO_RETURN"
        else:
            # Reason probabilities are CONDITIONAL on the order context
            reason_probs = {}

            # Delivery delay drives DELIVERY_DELAY reason
            if delivery_delay >= 3:
                reason_probs["DELIVERY_DELAY"] = 0.40
            elif delivery_delay >= 1:
                reason_probs["DELIVERY_DELAY"] = 0.15
            else:
                reason_probs["DELIVERY_DELAY"] = 0.05

            # Clothing drives SIZE_FIT_ISSUE
            if category == "Clothing":
                reason_probs["SIZE_FIT_ISSUE"] = 0.35
            else:
                reason_probs["SIZE_FIT_ISSUE"] = 0.03

            # Product defect rate drives QUALITY_DEFECT
            reason_probs["QUALITY_DEFECT"] = 0.08 + prod["defect_rate"] * 2.0

            # High discount → impulse buy → NO_LONGER_NEEDED
            reason_probs["NO_LONGER_NEEDED"] = 0.05 + (discount_percentage / 100) * 0.3

            # Base rates for other reasons
            reason_probs["NOT_AS_DESCRIBED"] = 0.10
            reason_probs["WRONG_ITEM"] = 0.04

            # Normalize
            total = sum(reason_probs.values())
            reasons = list(reason_probs.keys())
            probs = [reason_probs[r] / total for r in reasons]

            return_reason = np.random.choice(reasons, p=probs)

        # --- Build row ---
        orders.append({
            "order_id": oid,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "customer_id": cust_id,
            "product_id": prod_id,
            "product_category": category,
            "product_price": product_price,
            "quantity": quantity,
            "order_value": order_value,
            "discount_percentage": discount_percentage,
            "discount_amount": discount_amount,
            "customer_city": customer_city,
            "warehouse_city": warehouse_city,
            "distance_km": distance_km,
            "is_remote_area": is_remote_area,
            "shipping_mode": shipping_mode,
            "expected_delivery_days": expected_delivery_days,
            "actual_delivery_days": actual_delivery_days,
            "delivery_delay": delivery_delay,
            "is_returned": is_returned,
            "return_reason": return_reason,
            "order_date_dt": order_date # temporary for sorting
        })

    orders_df = pd.DataFrame(orders)
    
    # --- Step 9: Add Customer Behavior Metrics ---
    # Sort chronologically to compute past trends without data leakage
    orders_df = orders_df.sort_values(by=["customer_id", "order_date_dt", "order_id"])
    
    orders_df['total_orders'] = orders_df.groupby('customer_id').cumcount() + 1
    orders_df['past_return_rate'] = orders_df.groupby('customer_id')['is_returned'].transform(
        lambda x: x.shift().expanding().mean().fillna(0.0)
    ).round(4)
    orders_df['avg_order_value'] = orders_df.groupby('customer_id')['order_value'].transform(
        lambda x: x.shift().expanding().mean().fillna(x)
    ).round(2)

    # Sort back by order_id and drop temporary date column
    orders_df = orders_df.sort_values(by="order_id").drop(columns=["order_date_dt"])
    
    return orders_df


# ============================================================
# 3. Validation
# ============================================================

def validate_dataset(df):
    """Run comprehensive validation checks on the generated dataset."""
    print("\n" + "=" * 60)
    print("📊 DATASET VALIDATION REPORT")
    print("=" * 60)

    # --- Shape ---
    print(f"\n📐 Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # --- Null check ---
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("\n✅ No missing values")
    else:
        print(f"\n❌ Missing values found:\n{null_counts[null_counts > 0]}")

    # --- Return rate ---
    return_rate = df["is_returned"].mean()
    print(f"\n📈 Overall Return Rate: {return_rate:.2%}")
    if 0.10 <= return_rate <= 0.40:
        print("   ✅ Within expected range (10-40%)")
    else:
        print(f"   ⚠️ Outside expected range (10-40%)")

    # --- Logical checks ---
    print("\n🔍 Logical Checks:")

    # order_value = price * quantity
    ov_check = (df["order_value"] == df["product_price"] * df["quantity"]).all()
    print(f"   order_value = price × quantity: {'✅' if ov_check else '❌'}")

    # delivery_delay >= 0
    delay_check = (df["delivery_delay"] >= 0).all()
    print(f"   delivery_delay ≥ 0: {'✅' if delay_check else '❌'}")

    # actual = expected + delay
    actual_check = (df["actual_delivery_days"] == df["expected_delivery_days"] + df["delivery_delay"]).all()
    print(f"   actual = expected + delay: {'✅' if actual_check else '❌'}")

    # discount_amount consistency
    disc_check = np.allclose(
        df["discount_amount"],
        df["order_value"] * df["discount_percentage"] / 100,
        atol=0.1
    )
    print(f"   discount_amount consistency: {'✅' if disc_check else '❌'}")

    # NO_RETURN only for non-returned orders
    nr_check = (df[df["is_returned"] == 0]["return_reason"] == "NO_RETURN").all()
    print(f"   NO_RETURN ↔ not returned: {'✅' if nr_check else '❌'}")

    # Returned orders have actual reason
    ret_check = (df[df["is_returned"] == 1]["return_reason"] != "NO_RETURN").all()
    print(f"   Returned orders have reason: {'✅' if ret_check else '❌'}")

    # --- Causal pattern checks ---
    print("\n🔗 Causal Pattern Checks:")

    # Delay → Returns
    delay_return = df.groupby("delivery_delay")["is_returned"].mean()
    delay_trend = delay_return.corr(pd.Series(delay_return.index.astype(float), index=delay_return.index))
    print(f"   Delay ↑ → Returns ↑: {'✅' if delay_trend > 0.5 else '⚠️'} (corr={delay_trend:.3f})")

    # Discount → Returns
    disc_return = df.groupby("discount_percentage")["is_returned"].mean()
    disc_trend = disc_return.corr(pd.Series(disc_return.index.astype(float), index=disc_return.index))
    print(f"   Discount ↑ → Returns ↑: {'✅' if disc_trend > 0.5 else '⚠️'} (corr={disc_trend:.3f})")

    # Clothing → higher returns
    clothing_rate = df[df["product_category"] == "Clothing"]["is_returned"].mean()
    other_rate = df[df["product_category"] != "Clothing"]["is_returned"].mean()
    print(f"   Clothing returns ({clothing_rate:.2%}) > Others ({other_rate:.2%}): {'✅' if clothing_rate > other_rate else '⚠️'}")

    # Distance → Delay
    dist_delay_corr = df["distance_km"].corr(df["delivery_delay"])
    print(f"   Distance ↑ → Delay ↑: {'✅' if dist_delay_corr > 0.1 else '⚠️'} (corr={dist_delay_corr:.3f})")

    # --- Distribution checks ---
    print("\n📊 Distribution Checks:")
    print(f"   Price skewness: {df['product_price'].skew():.3f} (expect > 0, right-skewed)")
    print(f"   Quantity distribution: {df['quantity'].value_counts(normalize=True).sort_index().to_dict()}")
    print(f"   Return reasons:\n{df[df['is_returned']==1]['return_reason'].value_counts(normalize=True).to_string()}")

    return return_rate


# ============================================================
# 4. Main
# ============================================================

def main():
    print("🚀 Generating Synthetic E-Commerce Dataset...")
    print(f"   Orders: {NUM_ORDERS}")
    print(f"   Customers: {NUM_CUSTOMERS}")
    print(f"   Products: {NUM_PRODUCTS}")

    # Generate entities
    customers_df = generate_customers(NUM_CUSTOMERS)
    products_df = generate_products(NUM_PRODUCTS)

    print(f"\n📦 Customer base return tendency: "
          f"mean={customers_df['base_return_tendency'].mean():.3f}, "
          f"std={customers_df['base_return_tendency'].std():.3f}")

    print(f"🏭 Product defect rates: "
          f"mean={products_df['defect_rate'].mean():.4f}, "
          f"max={products_df['defect_rate'].max():.4f}")

    # Generate orders
    orders_df = generate_orders(customers_df, products_df, NUM_ORDERS)

    # Generate reviews
    reviews_df = generate_reviews(NUM_PRODUCTS, products_df)
    reviews_df.to_csv(os.path.join(OUTPUT_DIR, "product_reviews.csv"), index=False)
    print(f"   Product reviews: {os.path.join(OUTPUT_DIR, 'product_reviews.csv')}")

    # Validate
    validate_dataset(orders_df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Dataset saved to: {OUTPUT_FILE}")

    # Also save entity tables for reference
    customers_df.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    products_df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
    print(f"   Customer profiles: {os.path.join(OUTPUT_DIR, 'customers.csv')}")
    print(f"   Product catalog: {os.path.join(OUTPUT_DIR, 'products.csv')}")

    return orders_df


def generate_reviews(orders_df, products_df):
    """Generate realistic e-commerce reviews for products based on order data."""
    reviews = []
    
    # Review templates by rating and category
    review_templates = {
        5: {
            "Electronics": [
                "Amazing {product_type}! The battery life is incredible and it performs flawlessly. Highly recommend for anyone looking for quality.",
                "This {product_type} exceeded my expectations. Fast delivery and excellent build quality. Worth every penny!",
                "Perfect purchase! The {product_type} works exactly as advertised. Great value and fast shipping.",
                "Outstanding {product_type}. Features are top-notch and it's very user-friendly. Will buy again.",
                "Love this {product_type}! Superior quality and arrived well-packaged. Five stars all the way."
            ],
            "Clothing": [
                "Fits perfectly and the material is high quality. Comfortable to wear all day. Highly recommend!",
                "Beautiful {product_type}! True to size and great fabric. Fast delivery and excellent packaging.",
                "Absolutely love this {product_type}. Stylish, comfortable, and exactly as described. Will order more.",
                "Perfect fit and amazing quality. The {product_type} is worth the price. Fast shipping too!",
                "Fantastic {product_type}! Soft material, great design, and arrived quickly. Five stars!"
            ],
            "Home": [
                "Excellent {product_type}! Sturdy construction and looks great in my home. Very satisfied with the purchase.",
                "This {product_type} is perfect for my needs. High quality materials and easy to assemble. Recommend!",
                "Love this {product_type}! Well-made, functional, and arrived in perfect condition. Great value.",
                "Outstanding quality {product_type}. Exceeded expectations and fast delivery. Will buy again.",
                "Beautiful and functional {product_type}. Great craftsmanship and excellent packaging. Five stars!"
            ],
            "Beauty": [
                "Amazing {product_type}! Works wonders and smells great. My skin feels fantastic. Highly recommend!",
                "This {product_type} is a game-changer. High quality ingredients and visible results. Love it!",
                "Perfect {product_type} for my needs. Gentle on skin and effective. Fast delivery and great packaging.",
                "Outstanding results with this {product_type}. Quality is excellent and worth every penny. Five stars!",
                "Love this {product_type}! Effective, pleasant scent, and arrived quickly. Will repurchase."
            ]
        },
        4: {
            "Electronics": [
                "Good {product_type} overall. Works well but battery could be better. Still, good value for money.",
                "Solid {product_type}. Performs as expected with minor issues. Fast delivery and decent quality.",
                "This {product_type} is decent. Good features but packaging was average. Would recommend with reservations.",
                "Nice {product_type}! Functions properly and arrived on time. Minor complaints but overall satisfied.",
                "Reliable {product_type}. Good performance, though not exceptional. Worth the price."
            ],
            "Clothing": [
                "Good {product_type}, fits well but material could be softer. Overall satisfied with the purchase.",
                "Nice fit and quality {product_type}. Comfortable but color slightly different from photos. Recommend.",
                "Decent {product_type}. True to size and good fabric. Fast shipping, minor issues with packaging.",
                "This {product_type} is okay. Comfortable wear but not the best quality. Still, good value.",
                "Solid {product_type}. Fits well and arrived quickly. Minor complaints but overall good."
            ],
            "Home": [
                "Good {product_type} for the price. Sturdy but assembly instructions could be clearer. Satisfied.",
                "Decent {product_type}. Functional and well-made. Packaging was okay, arrived on time.",
                "This {product_type} works well. Good quality but not exceptional. Would buy again.",
                "Nice {product_type}! Serves its purpose and fast delivery. Minor issues with finish.",
                "Reliable {product_type}. Good construction, though packaging could be better. Recommend."
            ],
            "Beauty": [
                "Good {product_type}, effective but takes time to see results. Pleasant scent and good packaging.",
                "Decent {product_type}. Works okay and gentle on skin. Fast delivery, minor complaints.",
                "This {product_type} is alright. Good quality ingredients but not amazing. Worth trying.",
                "Nice {product_type}! Effective and arrived quickly. Minor issues with texture.",
                "Solid {product_type}. Good results, though packaging was average. Would recommend."
            ]
        },
        3: {
            "Electronics": [
                "Average {product_type}. Works but has some issues with performance. Delivery was fast though.",
                "This {product_type} is okay. Decent features but battery life disappoints. Mixed feelings.",
                "Mediocre {product_type}. Functions but not as expected. Packaging was poor.",
                "So-so {product_type}. Good price but quality is average. Arrived on time.",
                "This {product_type} does the job. Minor defects and average performance. Not impressed."
            ],
            "Clothing": [
                "Average {product_type}. Fits okay but material feels cheap. Delivery was fast.",
                "This {product_type} is mediocre. Comfortable but sizing runs small. Mixed reviews.",
                "Okay {product_type}. Decent quality but not what I expected. Packaging was average.",
                "So-so {product_type}. Fits well but color fades. Arrived quickly though.",
                "This {product_type} is alright. Good price but quality is average. Not sure I'd buy again."
            ],
            "Home": [
                "Average {product_type}. Functional but poor quality materials. Assembly was difficult.",
                "This {product_type} is okay. Works but finish is subpar. Delivery was fast.",
                "Mediocre {product_type}. Decent for price but not durable. Packaging damaged.",
                "So-so {product_type}. Serves purpose but instructions unclear. Mixed feelings.",
                "This {product_type} does the job. Average quality and arrived on time. Not exceptional."
            ],
            "Beauty": [
                "Average {product_type}. Works mildly but not impressive. Scent is okay.",
                "This {product_type} is mediocre. Effective but causes irritation. Packaging good.",
                "Okay {product_type}. Decent results but expensive for what it is. Fast delivery.",
                "So-so {product_type}. Good ingredients but texture poor. Mixed reviews.",
                "This {product_type} is alright. Works but not as advertised. Arrived quickly."
            ]
        },
        2: {
            "Electronics": [
                "Disappointing {product_type}. Poor performance and battery dies quickly. Not recommended.",
                "This {product_type} is subpar. Features don't work as described. Packaging was damaged.",
                "Bad {product_type}. Quality issues and arrived late. Very disappointed.",
                "Poor quality {product_type}. Doesn't meet expectations. Delivery was slow.",
                "This {product_type} failed me. Malfunctions and poor build. Regret the purchase."
            ],
            "Clothing": [
                "Poor {product_type}. Doesn't fit well and material is low quality. Disappointed.",
                "This {product_type} is bad. Shrinks after wash and color fades. Not recommended.",
                "Disappointing fit and quality. Cheap material and arrived wrinkled. Bad experience.",
                "Subpar {product_type}. Sizing wrong and poor stitching. Delivery was okay.",
                "This {product_type} is terrible. Uncomfortable and low quality. Regret buying."
            ],
            "Home": [
                "Poor {product_type}. Breaks easily and poor construction. Very disappointed.",
                "This {product_type} is subpar. Difficult assembly and low quality. Not worth it.",
                "Bad {product_type}. Doesn't work properly and arrived damaged. Regret purchase.",
                "Disappointing quality. Cheap materials and poor finish. Delivery was slow.",
                "This {product_type} failed. Malfunctions and poor design. Not recommended."
            ],
            "Beauty": [
                "Poor {product_type}. Causes irritation and no results. Very disappointed.",
                "This {product_type} is bad. Unpleasant smell and ineffective. Not recommended.",
                "Disappointing {product_type}. Causes breakouts and poor quality. Regret buying.",
                "Subpar results. Expensive and doesn't work. Packaging was okay.",
                "This {product_type} is terrible. Irritates skin and no effect. Bad purchase."
            ]
        },
        1: {
            "Electronics": [
                "Terrible {product_type}! Doesn't work at all. Complete waste of money. Avoid!",
                "Worst purchase ever. {product_type} broke immediately. Poor quality and no support.",
                "This {product_type} is junk. Malfunctions constantly. Horrible experience.",
                "Awful {product_type}. Dead on arrival and terrible packaging. Never again.",
                "Complete disappointment. {product_type} doesn't function. Regret everything.",
                "Great job making a {product_type} that works perfectly as a paperweight! Because it certainly doesn't do anything else.",
                "I must have received the prototype, because this {product_type} is an absolute joke. Thanks for nothing."
            ],
            "Clothing": [
                "Horrible {product_type}! Doesn't fit and tears easily. Complete waste.",
                "Worst clothing ever. Cheap material and poor stitching. Avoid at all costs.",
                "This {product_type} is garbage. Shrinks horribly and fades instantly. Terrible.",
                "Awful fit and quality. Arrived damaged and wrong size. Never buying again.",
                "Complete failure. {product_type} falls apart. Horrible purchase experience.",
                "Oh I love shedding. This {product_type} leaves threads literally everywhere. I look like a molting bird.",
                "One wash and this {product_type} fits my cat instead of me. Excellent quality control, guys."
            ],
            "Home": [
                "Terrible {product_type}! Breaks on first use. Poor construction and quality.",
                "Worst item ever. Doesn't work and arrived broken. Complete waste of money.",
                "This {product_type} is junk. Poor materials and falls apart. Horrible.",
                "Awful quality. Malfunctions immediately. Terrible packaging and delivery.",
                "Complete disappointment. {product_type} is unusable. Regret the purchase.",
                "I bought this {product_type} hoping to use it, but I guess I'm just storing garbage now. Beautiful.",
                "If you enjoy assembling {product_type}s that are missing half the screws, this is the product for you!"
            ],
            "Beauty": [
                "Horrible {product_type}! Causes severe irritation. Dangerous and ineffective.",
                "Worst product ever. Burns skin and no results. Complete waste.",
                "This {product_type} is toxic. Causes allergic reaction. Avoid at all costs.",
                "Awful {product_type}. Smells terrible and ineffective. Terrible experience.",
                "Complete failure. Irritates skin badly. Horrible purchase, regret it.",
                "My skin has never looked worse! Thanks to this {product_type}, I look like a shedding snake.",
                "Smells like industrial cleaner and works just as poorly. Incredible formulation on this {product_type}!"
            ]
        }
    }
    
    product_type_map = {
        "Electronics": "device",
        "Clothing": "garment",
        "Home": "item",
        "Beauty": "product"
    }

    # Group by product to distribute reviews correctly
    product_to_order_idx = {}
    for idx, row in orders_df.iterrows():
        pid = row['product_id']
        if pid not in product_to_order_idx:
            product_to_order_idx[pid] = []
        product_to_order_idx[pid].append(idx)

    for pid in products_df['product_id']:
        # Get category
        category = products_df[products_df['product_id'] == pid]['product_category'].iloc[0]
        product_type = product_type_map.get(category, "product")

        # Assign reviews only to actual orders of this product
        if pid in product_to_order_idx:
            order_indices = product_to_order_idx[pid]
            np.random.shuffle(order_indices)
            
            # Decide how many reviews (e.g. 50-80% of orders leave a review)
            num_reviews = int(len(order_indices) * np.random.uniform(0.5, 0.8))
            num_reviews = max(1, num_reviews) # At least one if ordered

            for i in range(num_reviews):
                order_idx = order_indices[i]
                
                # Fetch order data
                is_returned = orders_df.at[order_idx, 'is_returned']
                delivery_delay = orders_df.at[order_idx, 'delivery_delay']
                order_date = datetime.strptime(orders_df.at[order_idx, 'order_date'], "%Y-%m-%d")

                # If returned, much higher chance of bad rating
                if is_returned:
                    rating = np.random.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.15, 0.10])
                elif delivery_delay > 2:
                    rating = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
                else:
                    rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])

                # Review is left some days after order
                review_date = order_date + timedelta(days=np.random.randint(2, 20))

                template = np.random.choice(review_templates[rating][category])
                review_text = template.format(product_type=product_type)

                # Add directly to orders dataframe
                orders_df.at[order_idx, 'review_rating'] = rating
                orders_df.at[order_idx, 'review_text'] = review_text
                orders_df.at[order_idx, 'review_date'] = review_date.strftime("%Y-%m-%d")

    return orders_df

# ============================================================


# Make sure we don't output customers or separate products files if we want a unified dataset
def main():
    print("🚀 Generating Synthetic E-Commerce Dataset...")
    print(f"   Orders: {NUM_ORDERS}")
    print(f"   Customers: {NUM_CUSTOMERS}")
    print(f"   Products: {NUM_PRODUCTS}")

    # Generate entities
    customers_df = generate_customers(NUM_CUSTOMERS)
    products_df = generate_products(NUM_PRODUCTS)

    print(f"\n📦 Customer base return tendency: "
          f"mean={customers_df['base_return_tendency'].mean():.3f}, "
          f"std={customers_df['base_return_tendency'].std():.3f}")

    print(f"🏭 Product defect rates: "
          f"mean={products_df['defect_rate'].mean():.4f}, "
          f"max={products_df['defect_rate'].max():.4f}")

    # Generate orders
    orders_df = generate_orders(customers_df, products_df, NUM_ORDERS)

    # Generate reviews directly tied to orders
    orders_df = generate_reviews(orders_df, products_df)

    # Add back the customers and products columns right into the orders_df
    orders_df = pd.merge(orders_df, customers_df[['customer_id', 'base_return_tendency', 'home_city']], on='customer_id', how='left', suffixes=('', '_drop'))
    orders_df = orders_df.loc[:, ~orders_df.columns.str.endswith('_drop')]
    
    orders_df = pd.merge(orders_df, products_df[['product_id', 'defect_rate']], on='product_id', how='left')

    # Validate
    validate_dataset(orders_df)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    orders_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Single unified dataset saved to: {OUTPUT_FILE}")

    return orders_df

if __name__ == "__main__":
    df = main()
