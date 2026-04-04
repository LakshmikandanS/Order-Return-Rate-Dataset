import pandas as pd
import re
import json

csv_path = 'Order-Return-Rate-Dataset/data/synthetic_ecommerce_orders.csv'

df = pd.read_csv(csv_path, parse_dates=['order_date','review_date'], dayfirst=False)

# Basic counts
total_orders = int(len(df))
reviews_count = int(df['review_text'].notna().sum())
missing_reviews = int(df['review_text'].isna().sum())

# Sarcasm detection using mathematical contradiction scores
if 'sarcasm_score' in df.columns:
    sarcastic_mask = df['sarcasm_score'] > 0.4
elif 'is_sarcastic' in df.columns:
    sarcastic_mask = df['is_sarcastic'] == 1 
else:
    sarcastic_mask = pd.Series(False, index=df.index)

sarcastic_count = int(sarcastic_mask.sum())

sarcastic_pct_reviews = float(sarcastic_count / reviews_count * 100) if reviews_count else 0.0
sarcastic_pct_all = float(sarcastic_count / total_orders * 100) if total_orders else 0.0

# Per-category return rates
# Ensure is_returned is numeric
try:
    df['is_returned'] = pd.to_numeric(df['is_returned'], errors='coerce').fillna(0).astype(int)
except Exception:
    df['is_returned'] = df['is_returned'].apply(lambda x: 1 if str(x).strip() in ['1','True','TRUE','true'] else 0)

per_category = df.groupby('product_category').agg(
    total_orders=('order_id','count'),
    returns=('is_returned', 'sum')
).reset_index()
per_category['return_rate'] = (per_category['returns'] / per_category['total_orders']) * 100
per_category_sorted = per_category.sort_values('return_rate', ascending=False)

# Top return reasons
top_reasons = df[df['is_returned']==1]['return_reason'].value_counts(normalize=True) * 100

# Missingness per column
missing_per_column = df.isna().sum().to_dict()

# Price skewness and quantity distribution
price_skew = float(df['product_price'].skew())
qty_dist = (df['quantity'].value_counts(normalize=True) * 100).to_dict()
qty_dist = {int(k): float(v) for k,v in qty_dist.items()}

out = {
    'total_orders': total_orders,
    'reviews_count': reviews_count,
    'missing_reviews': missing_reviews,
    'sarcastic_count': sarcastic_count,
    'sarcastic_pct_reviews': round(sarcastic_pct_reviews, 4),
    'sarcastic_pct_all_orders': round(sarcastic_pct_all, 4),
    'per_category': per_category_sorted.to_dict(orient='records'),
    'top_reasons_pct': top_reasons.to_dict(),
    'missing_per_column': {k:int(v) for k,v in missing_per_column.items()},
    'price_skew': round(price_skew, 4),
    'quantity_distribution_pct': qty_dist
}

print(json.dumps(out, indent=2))
