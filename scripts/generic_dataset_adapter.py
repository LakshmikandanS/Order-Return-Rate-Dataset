import pandas as pd
import json
import random
import os
import numpy as np

def adapt_dataset(input_file, output_file, column_mapping, dataset_name="External", limit=None):
    """
    Generalized generic adapter to map external datasets (JSON or CSV) to the Supply Chain Project schema.
    If columns are missing from the mapping, it generates sensible standard structural defaults/distributions.
    """
    print(f"Loading data from {input_file}...")
    
    # 1. Load Data
    if input_file.endswith('.json'):
        if limit:
            records = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= limit: break
                    records.append(json.loads(line))
            df = pd.DataFrame(records)
        else:
            df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file, nrows=limit)
        
    print(f"Loaded {len(df)} rows. Mapping columns...")
    
    # 2. Map existing columns
    df_mapped = pd.DataFrame()
    for ext_col, std_col in column_mapping.items():
        if ext_col in df.columns:
            df_mapped[std_col] = df[ext_col]
            
    # 3. Define Standard Target Schema
    standard_columns = [
        'order_id', 'customer_id', 'product_id', 'order_date', 'review_date',
        'review_text', 'review_rating', 'product_category', 'product_price',
        'quantity', 'order_value', 'discount_percentage', 'discount_amount',
        'customer_city', 'warehouse_city', 'home_city', 'distance_km',
        'is_remote_area', 'shipping_mode', 'expected_delivery_days',
        'actual_delivery_days', 'delivery_delay', 'is_returned', 'return_reason',
        'total_orders', 'past_return_rate', 'avg_order_value',
        'base_return_tendency', 'defect_rate'
    ]
    
    # Set seeds for reproducible mappings
    random.seed(42)
    np.random.seed(42)
    length = len(df_mapped)
    
    # 4. Synthesize Missing Standard Columns
    
    # IDs
    if 'order_id' not in df_mapped:
        df_mapped['order_id'] = [f"{dataset_name.upper()}_{i}" for i in range(length)]
    if 'customer_id' not in df_mapped:
        df_mapped['customer_id'] = [f"CUST_{random.randint(1000, 9999)}" for _ in range(length)]
    if 'product_id' not in df_mapped:
        df_mapped['product_id'] = [f"PROD_{random.randint(100, 999)}" for _ in range(length)]
        
    # Dates
    if 'order_date' in df_mapped:
        # If numeric, assume unix timestamp
        if pd.api.types.is_numeric_dtype(df_mapped['order_date']):
            df_mapped['order_date'] = pd.to_datetime(df_mapped['order_date'], unit='s', errors='coerce')
        else:
            df_mapped['order_date'] = pd.to_datetime(df_mapped['order_date'], errors='coerce')
    else:
        df_mapped['order_date'] = pd.Timestamp('2023-01-01')
        
    if 'review_date' not in df_mapped:
        df_mapped['review_date'] = df_mapped['order_date'] + pd.to_timedelta(np.random.randint(1, 15, length), unit='d')
        
    # Product / Financials
    if 'product_category' not in df_mapped:
        df_mapped['product_category'] = dataset_name
    if 'product_price' not in df_mapped:
        df_mapped['product_price'] = np.round(np.random.uniform(5.0, 150.0, length), 2)
    if 'quantity' not in df_mapped:
        df_mapped['quantity'] = np.random.choice([1, 2, 3], length, p=[0.7, 0.2, 0.1])
    if 'order_value' not in df_mapped:
        df_mapped['order_value'] = df_mapped['product_price'] * df_mapped['quantity']
    if 'discount_percentage' not in df_mapped:
        df_mapped['discount_percentage'] = np.random.choice([0.0, 0.05, 0.1, 0.2], length, p=[0.6, 0.2, 0.1, 0.1])
    if 'discount_amount' not in df_mapped:
        df_mapped['discount_amount'] = df_mapped['order_value'] * df_mapped['discount_percentage']
        
    # Logistics
    if 'distance_km' not in df_mapped:
        df_mapped['distance_km'] = np.round(np.random.uniform(10.0, 1500.0, length), 1)
    if 'is_remote_area' not in df_mapped:
         df_mapped['is_remote_area'] = np.random.choice([0, 1], length, p=[0.85, 0.15])
    if 'shipping_mode' not in df_mapped:
        df_mapped['shipping_mode'] = np.random.choice(['Standard', 'Express', 'Same-Day'], length)
        
    if 'expected_delivery_days' not in df_mapped:
        df_mapped['expected_delivery_days'] = np.random.randint(2, 8, length)
    
    if 'actual_delivery_days' not in df_mapped:
        delay_variance = np.random.choice([0, 0, 0, 1, 2, 3, -1], length)
        df_mapped['actual_delivery_days'] = (df_mapped['expected_delivery_days'] + delay_variance).clip(lower=0)
        
    if 'delivery_delay' not in df_mapped:
        df_mapped['delivery_delay'] = (df_mapped['actual_delivery_days'] - df_mapped['expected_delivery_days']).clip(lower=0)
        
    # Reviews
    if 'review_text' not in df_mapped:
        df_mapped['review_text'] = 'NO_REVIEW'
    else:
        df_mapped['review_text'] = df_mapped['review_text'].fillna('NO_REVIEW')
        
    if 'review_rating' not in df_mapped:
        df_mapped['review_rating'] = 3.0
        
    # Returns / Defects
    if 'is_returned' not in df_mapped:
        def infer_return(row):
            text = str(row['review_text']).lower()
            # Basic NLP sentiment inference for returns
            if 'return' in text or 'sending back' in text or 'refund' in text or 'waste of money' in text:
                return 1
            if row['review_rating'] <= 2 and random.random() < 0.3:
                return 1
            return 0
        df_mapped['is_returned'] = df_mapped.apply(infer_return, axis=1)
        
    if 'return_reason' not in df_mapped:
        df_mapped['return_reason'] = df_mapped['is_returned'].apply(lambda x: 'QUALITY_DEFECT' if x else 'NO_RETURN')
        
    if 'defect_rate' not in df_mapped:
        df_mapped['defect_rate'] = np.round(np.random.uniform(0.01, 0.1, length), 3)
        
    # Customer History
    if 'total_orders' not in df_mapped:
         df_mapped['total_orders'] = np.random.randint(1, 10, length)
    if 'avg_order_value' not in df_mapped:
         df_mapped['avg_order_value'] = df_mapped['order_value']
    if 'past_return_rate' not in df_mapped:
         df_mapped['past_return_rate'] = np.round(np.random.uniform(0.0, 0.4, length), 2)
    if 'base_return_tendency' not in df_mapped:
         df_mapped['base_return_tendency'] = np.round(np.random.uniform(0.0, 0.3, length), 2)
         
    # Cities/Locations
    for c in ['customer_city', 'warehouse_city', 'home_city']:
        if c not in df_mapped:
            df_mapped[c] = 'Unknown'
            
    # Ensure correct column order & missing safeguards
    for col in standard_columns:
        if col not in df_mapped.columns:
            df_mapped[col] = 0

    df_mapped = df_mapped[standard_columns]
    
    # Create Data Dir if necessary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_mapped.to_csv(output_file, index=False)
    print(f"✅ Successfully saved {len(df_mapped)} mapped records to {output_file}")
    return df_mapped


if __name__ == '__main__':
    # Usage Example: Map the Amazon Fashion JSON
    amazon_mapping = {
        'reviewerID': 'customer_id',
        'asin': 'product_id',
        'reviewText': 'review_text',
        'overall': 'review_rating',
        'unixReviewTime': 'order_date'
    }
    
    input_path = '../data/AMAZON_FASHION.json'
    output_path = '../data/external_amazon_fashion_mapped.csv'
    
    if os.path.exists(input_path):
        adapt_dataset(
            input_file=input_path,
            output_file=output_path,
            column_mapping=amazon_mapping,
            dataset_name='Amazon_Fashion',
            limit=10000
        )
    else:
        print(f"Skipped Amazon Fashion generic adapt - File not found at {input_path}")