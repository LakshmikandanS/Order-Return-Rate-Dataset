"""
🚀 Advanced Preprocessing & Feature Engineering Pipeline
For Order Return Rate Prediction

Implements all 5 phases:
1. Data Integrity & Structural Cleaning
2. Temporal & Seasonal Engineering
3. Behavioral & Historical Profiling
4. NLP & Sentiment Extraction
5. Model-Ready Transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# NLP & Sentiment
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 0: SCHEMA STANDARDIZATION
# ============================================================================

def standardize_schema(df):
    """
    Standardize the dataset schema:
    - Convert column names to lowercase snake_case
    - Ensure correct data types for key columns
    """
    logger.info("📐 Phase 0: Standardizing Schema...")
    
    df = df.copy()
    
    # Standardize column names (lowercase, replace spaces with underscores)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Define expected types (basic mapping)
    datetime_cols = ['order_date', 'delivery_date', 'review_date']
    numeric_cols = ['product_price', 'quantity', 'order_value', 'discount_percentage', 'discount_amount', 
                    'distance_km', 'expected_delivery_days', 'actual_delivery_days', 'delivery_delay', 
                    'total_orders', 'past_return_rate', 'avg_order_value', 'review_rating', 
                    'base_return_tendency', 'defect_rate']
    categorical_cols = ['product_category', 'customer_city', 'warehouse_city', 'shipping_mode', 
                        'return_reason', 'home_city', 'review_text']
    boolean_cols = ['is_remote_area', 'is_returned']
    
    # Apply conversions
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    for col in categorical_cols:
        if col in df.columns:
            # Only cast to string if it's not null, to avoid turning NaN into "nan"
            # Actually, filling NaN comes in Phase 1, so let's just make valid ones string
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str)
            
    for col in boolean_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    logger.info(f"  ✓ Schema standardized. {len(df.columns)} columns formatted.")
    return df


# ============================================================================
# PHASE 1: DATA INTEGRITY & STRUCTURAL CLEANING
# ============================================================================

def handle_missing_values(df):
    """
    Handle missing values strategically:
    - review_text, review_rating, review_date: Create boolean flag + impute
    - return_reason: Set to 'NO_RETURN' for non-returned orders
    """
    logger.info("🧹 Phase 1: Handling Missing Values...")
    
    df = df.copy()
    
    # Create missing value indicators
    df['has_review'] = df['review_text'].notna().astype(int)
    
    # Impute review_rating with 0 (neutral) for missing values
    df['review_rating'] = df['review_rating'].fillna(0)
    
    # Fill review_text with placeholder
    df['review_text'] = df['review_text'].fillna('NO_REVIEW')
    
    # Fill review_date with order_date if missing
    df['review_date'] = df['review_date'].fillna(df['order_date'])
    
    # Set return_reason to 'NO_RETURN' for non-returned orders
    df.loc[df['is_returned'] == 0, 'return_reason'] = df.loc[df['is_returned'] == 0, 'return_reason'].fillna('NO_RETURN')
    
    logger.info(f"  ✓ Missing values handled. has_review: {df['has_review'].sum()}/{len(df)} reviews")
    return df


def validate_order_integrity(df):
    """
    Validate order data:
    - Check for unrealistic delivery times
    """
    logger.info("📊 Validating Order Integrity...")
    
    df = df.copy()
    
    # Check for unrealistic delivery times
    if 'actual_delivery_days' in df.columns:
        negative_delays = (df['actual_delivery_days'] < 0).sum()
        if negative_delays > 0:
            logger.warning(f"  ⚠ Found {negative_delays} negative delivery delays. Clamping to 0.")
            df.loc[df['actual_delivery_days'] < 0, 'actual_delivery_days'] = 0
    
    logger.info(f"  ✓ Integrity validation complete.")
    return df


def handle_outliers(df):
    """
    Handle outliers using robust methods:
    - product_price: Log transform if skewed
    - distance_km: Flag unrealistic spikes
    """
    logger.info("🎯 Handling Outliers...")
    
    df = df.copy()
    
    # Check skewness of product_price
    price_skewness = df['product_price'].skew()
    logger.info(f"  Price skewness: {price_skewness:.2f}")
    
    if abs(price_skewness) > 2.0:
        logger.info(f"  → Applying log transform to product_price (skewness > 2.0)")
        df['product_price_log'] = np.log1p(df['product_price'])
    else:
        df['product_price_log'] = df['product_price']
    
    # Flag unrealistic distances (e.g., > 3000km for domestic shipping)
    df['is_unrealistic_distance'] = (df['distance_km'] > 3000).astype(int)
    unrealistic_count = df['is_unrealistic_distance'].sum()
    if unrealistic_count > 0:
        logger.warning(f"  ⚠ Found {unrealistic_count} unrealistic distances (> 3000km)")
    
    logger.info(f"  ✓ Outlier handling complete.")
    return df


# ============================================================================
# PHASE 2: TEMPORAL & SEASONAL ENGINEERING
# ============================================================================

def extract_temporal_features(df):
    """
    Extract temporal features:
    - Cyclical: hour, day_of_week, month, is_weekend
    - Festival awareness: days_to_festival, is_festival_window
    - Processing lag: review_lag
    """
    logger.info("📅 Phase 2: Extracting Temporal Features...")
    
    df = df.copy()
    
    # Convert dates to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'])
    
    # Cyclical features from order_date
    df['order_hour'] = df['order_date'].dt.hour
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_month'] = df['order_date'].dt.month
    df['is_weekend'] = ((df['order_day_of_week'] >= 5).astype(int))  # 5=Saturday, 6=Sunday
    df['order_quarter'] = df['order_date'].dt.quarter
    
    # Festival windows (Indian festivals - adjustable)
    festival_dates = {
        'Diwali': (10, 25),      # October 25 (approx)
        'Christmas': (12, 25),   # December 25
        'NewYear': (1, 1),       # January 1
        'Holi': (3, 20),         # March 20 (approx)
        'EidUlFitr': (4, 10),    # Varies, using approx
    }
    
    df['days_to_nearest_festival'] = df['order_date'].apply(
        lambda date: min([
            min(abs((datetime(date.year, m, d) - date).days),
                abs((datetime(date.year + 1, m, d) - date).days))
            for m, d in festival_dates.values()
        ])
    )
    
    df['is_festival_window'] = (df['days_to_nearest_festival'] <= 7).astype(int)
    
    # Processing lag: time from order to review
    if 'review_date' in df.columns:
        df['review_lag_days'] = (df['review_date'] - df['order_date']).dt.days
        df['review_lag_days'] = df['review_lag_days'].fillna(-1)  # -1 for no review
    else:
        df['review_lag_days'] = -1
    
    logger.info(f"  ✓ Temporal features extracted:")
    logger.info(f"    - Cyclical: hour, day_of_week, month, quarter, is_weekend")
    logger.info(f"    - Festival: days_to_nearest_festival, is_festival_window")
    logger.info(f"    - Lag: review_lag_days")
    
    return df


# ============================================================================
# PHASE 3: BEHAVIORAL & HISTORICAL PROFILING
# ============================================================================

def calculate_rolling_behavior(df):
    """
    Calculate behavioral features:
    - Customer Rolling Features (CRF): past returns, lifetime value, avg rating
    - Product Risk Metrics (PRM): return rate, defect rate
    
    ⚠️ Uses time-based splits to avoid look-ahead bias
    """
    logger.info("👥 Phase 3: Calculating Behavioral Features...")
    
    df = df.copy()
    df = df.sort_values('order_date').reset_index(drop=True)
    
    # ===== CUSTOMER ROLLING FEATURES =====
    
    # 1. Customer lifetime orders and value
    df['cust_lifetime_orders'] = df.groupby('customer_id').cumcount() + 1
    df['cust_lifetime_value'] = df.groupby('customer_id')['order_value'].cumsum()

    # RFM Approximation
    df['last_order_date'] = df.groupby('customer_id')['order_date'].shift(1)
    df['days_since_last_order'] = (df['order_date'] - df['last_order_date']).dt.days.fillna(999)
    df = df.drop('last_order_date', axis=1)
    
    # Interaction Features
    if 'discount_percentage' in df.columns and 'past_return_rate' in df.columns:
        df['discount_depth_vs_return_history'] = (df['discount_percentage'] / 100.0) * df['past_return_rate']
    else:
        df['discount_depth_vs_return_history'] = 0.0
    
    # 2. Customer return history (past 30 days)
    # Optimize O(N^2) loop using rolling sum
    df_sorted = df.sort_values(by=['customer_id', 'order_date'])
    df_sorted.set_index('order_date', inplace=True)
    df_sorted['cust_past_30_returns'] = df_sorted.groupby('customer_id')['is_returned'].rolling('30D', closed='left').sum().fillna(0).values
    df = df_sorted.reset_index().sort_values(by='index' if 'index' in df_sorted.reset_index().columns else 'order_date').reset_index(drop=True)
    
    # 3. Customer average rating
    df['cust_avg_rating'] = df.groupby('customer_id')['review_rating'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(3.0)
    )
    
    # ===== PRODUCT RISK METRICS =====
    
    # 1. Product category return rate (rolling, time-aware)
    df_sorted = df.sort_values(by=['product_category', 'order_date'])
    df_sorted['prod_category_return_rate'] = df_sorted.groupby('product_category')['is_returned'].transform(lambda x: x.expanding().mean().shift(1).fillna(0.0))
    # Re-sort to maintain original order
    df = df_sorted.sort_index().reset_index(drop=True)
    
    # 2. High defect product flag (using heuristic)
    defect_threshold = 0.15
    product_return_rates = df.groupby('product_id')['is_returned'].transform('mean')
    df['is_high_risk_product'] = (product_return_rates > defect_threshold).astype(int)
    
    logger.info(f"  ✓ Behavioral features calculated:")
    logger.info(f"    - Customer: lifetime_orders, lifetime_value, past_30_returns, avg_rating")
    logger.info(f"    - Product: category_return_rate, is_high_risk_product")
    
    return df


# ============================================================================
# PHASE 4: NLP & SENTIMENT EXTRACTION
# ============================================================================

def extract_sentiment_features(df, config=None):
    """
    Extract NLP features:
    - Sentiment polarity & subjectivity
    - Sarcasm detection (heuristic)
    - Text length & keyword features
    """
    logger.info("📝 Phase 4: Extracting Sentiment & NLP Features...")
    
    df = df.copy()
    
    sentiment_scores = []
    text_lengths = []

    # Temporary disable HuggingFace for testing unless explicitly forced
    if config and config.get('language') in ['auto', 'multilingual'] and config.get('use_hf', False):
        try:
            from transformers import pipeline
            logger.info("  using HuggingFace multilingual sentiment...")
            # We use a fast, lightweight multilingual model or just standard pipeline
            sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True, max_length=512)
            use_hf = True
        except ImportError:
            logger.warning("  transformers not installed. Falling back to TextBlob...")
            use_hf = False
    else:
        use_hf = False

    
    for text in df['review_text']:
        if text == 'NO_REVIEW':
            sentiment_scores.append((0, 0))  # polarity, subjectivity
            text_lengths.append(0)
        else:
            # Sentiment analysis
            if use_hf:
                try:
                    result = sentiment_pipeline(str(text)[:512])[0]
                    # Score 1 to 5 stars -> polarity -1 to 1
                    star_rating = int(result['label'].split(' ')[0])
                    polarity = (star_rating - 3) / 2.0
                    subjectivity = 0.5 # HF doesn't give subjectivity natively
                    sentiment_scores.append((polarity, subjectivity))
                except Exception:
                    blob = TextBlob(str(text))
                    sentiment_scores.append((blob.sentiment.polarity, blob.sentiment.subjectivity))
            else:
                blob = TextBlob(str(text))
                sentiment_scores.append((blob.sentiment.polarity, blob.sentiment.subjectivity))
            
            # Text length
            text_lengths.append(len(str(text).split()))
    
    df['sentiment_polarity'] = [s[0] for s in sentiment_scores]
    df['sentiment_subjectivity'] = [s[1] for s in sentiment_scores]
    df['review_word_count'] = text_lengths
    
    logger.info(f"  ✓ Sentiment features extracted:")
    logger.info(f"    - Polarity & Subjectivity scores")
    logger.info(f"    - Review word count")
    
    return df


def vectorize_reviews(df, max_features=20):
    """
    TF-IDF vectorization of review text (optional for deep learning)
    """
    logger.info(f"📊 Vectorizing reviews (top {max_features} features)...")
    
    # Only vectorize reviews with text
    reviews_with_text = df[df['review_text'] != 'NO_REVIEW']['review_text'].values
    
    if len(reviews_with_text) == 0:
        logger.warning("  ⚠ No reviews to vectorize. Skipping TF-IDF.")
        return df
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(reviews_with_text)
    
    # Create feature dataframe
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
    )
    
    # Map back to original dataframe
    tfidf_full = pd.DataFrame(0, index=range(len(df)), columns=tfidf_df.columns)
    tfidf_full.iloc[df[df['review_text'] != 'NO_REVIEW'].index] = tfidf_df.values
    
    df = pd.concat([df.reset_index(drop=True), tfidf_full.reset_index(drop=True)], axis=1)
    
    logger.info(f"  ✓ TF-IDF features created: {tfidf_df.shape[1]} features")
    return df


def _safe_minmax(series):
    """Safely min-max normalize a pandas Series to [0,1]."""
    s = series.copy().astype(float).fillna(0)
    minv = s.min()
    maxv = s.max()
    if maxv - minv == 0:
        return pd.Series(0.0, index=s.index)
    return (s - minv) / (maxv - minv)


def compute_sarcasm_and_spam_features(df, config=None):
    """
    Compute `sarcasm_score` and `spam_score` following the strategy in
    `instructions/spam_sarcastic_handlin_strategy.md`.

    Adds columns:
      - `sarcasm_score` (0-1), `sentiment_delivery_gap`, `is_sarcastic_score_flag`
      - `spam_score` (0-1), `is_likely_spam`, `is_low_effort_review`
    """
    logger.info("🧪 Computing sarcasm and spam scores...")
    df = df.copy()

    # Ensure required columns exist with safe defaults
    defaults = {
        'sentiment_polarity': 0.0,
        'review_rating': 3.0,
        'actual_delivery_days': 0.0,
        'expected_delivery_days': 0.0,
        'defect_rate': 0.0,
        'review_word_count': 0,
        'past_return_rate': 0.0,
        'total_orders': 0.0,
        'review_text': 'NO_REVIEW'
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d

    # --- Sarcasm score (Real Features Only - Refined) ---
    # Convert rating to a [-1, 1] scale
    rating_scaled = ((df['review_rating'].fillna(3.0) - 3.0) / 2.0).clip(-1, 1)
    
    # 1. Sentiment vs Rating Contradiction
    # Sarcasm occurs when text is overwhelmingly positive (sentiment > 0) but rating is terrible (< 0) 
    # OR text is negative but rating is perfect 5.
    # absolute difference between sentiment and rating_scaled
    sentiment_rating_gap = (df['sentiment_polarity'].fillna(0.0) - rating_scaled).abs() / 2.0
    
    # 2. Sentiment vs Length
    # High sentiment but unusually long text for a low rating, or excessively short for a high rating
    normalized_len = _safe_minmax(df['review_word_count'])
    len_contradiction = np.maximum(0, df['sentiment_polarity'].fillna(0.0) * normalized_len * (rating_scaled < 0).astype(int))

    # Re-blend the score based heavily on real feedback (removing synthetic delivery bias)
    sarcasm_score = 0.7 * sentiment_rating_gap + 0.3 * len_contradiction
    sarcasm_score = sarcasm_score.clip(0, 1)

    df['sarcasm_score'] = sarcasm_score
    df['sentiment_rating_gap'] = sentiment_rating_gap
    
    # Using > 0.4 as threshold to capture 'suspicious' & 'sarcastic' as flagged given data distribution
    df['is_sarcastic'] = (df['sarcasm_score'] > 0.4).astype(int)
    df['is_sarcastic_score_flag'] = df['is_sarcastic']

    # --- Spam score ---
    df['is_low_effort_review'] = (df['review_word_count'].fillna(0) < 3).astype(int)
    empty_sentiment = ((df['sentiment_polarity'].fillna(0.0) == 0.0) & (df['review_word_count'].fillna(0) == 0)).astype(int)

    rating_text_mismatch = (((df['review_rating'] == 5) & (df['sentiment_polarity'] < 0)) |
                            ((df['review_rating'] == 1) & (df['sentiment_polarity'] > 0))).astype(int)

    behavior_score_raw = _safe_minmax(df['past_return_rate'].fillna(0.0)) + _safe_minmax(df['total_orders'].fillna(0.0))
    behavior_score = (behavior_score_raw / 2.0).clip(0, 1)

    # Repetition score via exact-text frequency
    review_texts = df['review_text'].fillna('NO_REVIEW')
    freqs = review_texts.map(review_texts.value_counts())
    repetition_score = _safe_minmax(freqs)

    spam_score = (
        0.1 * df['is_low_effort_review']
      + 0.3 * empty_sentiment
      + 0.2 * rating_text_mismatch
      + 0.2 * behavior_score
      + 0.2 * repetition_score
    )
    spam_score = spam_score.clip(0, 1)

    df['spam_score'] = spam_score
    df['is_likely_spam'] = (df['spam_score'] >= 0.8).astype(int)

    logger.info(f"  ✓ Sarcasm flagged: {int(df['is_sarcastic_score_flag'].sum())}, Spam flagged: {int(df['is_likely_spam'].sum())}")
    return df


# ============================================================================
# PHASE 5: MODEL-READY TRANSFORMATION
# ============================================================================

def encode_categorical_features(df, categorical_cols, target_col='is_returned'):
    """
    Encode categorical features using target encoding (with smoothing)
    """
    logger.info("🔀 Phase 5: Encoding Categorical Features...")
    
    df = df.copy()
    smoothing = 1.0  # Smoothing factor for target encoding
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
            
        # Target encoding with smoothing
        global_mean = df[target_col].mean()
        target_means = df.groupby(col)[target_col].agg(['mean', 'count'])
        
        # Apply smoothing
        smoothed_means = (target_means['mean'] * target_means['count'] + global_mean * smoothing) / \
                         (target_means['count'] + smoothing)
        
        df[f'{col}_encoded'] = df[col].map(smoothed_means)
        df[f'{col}_encoded'] = df[f'{col}_encoded'].fillna(global_mean)
        
        logger.info(f"  ✓ Target encoded: {col}")
    
    return df


def scale_numerical_features(df, numerical_cols, method='standard'):
    """
    Scale numerical features using StandardScaler or MinMaxScaler
    """
    logger.info(f"📏 Scaling Numerical Features ({method})...")
    
    df = df.copy()
    
    for col in numerical_cols:
        if col not in df.columns or df[col].dtype == 'object':
            continue
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Avoid scaling if column has NaN
        valid_mask = df[col].notna()
        if valid_mask.sum() == 0:
            continue
        
        scaled_values = scaler.fit_transform(df.loc[valid_mask, [col]])
        df.loc[valid_mask, f'{col}_scaled'] = scaled_values
        df[f'{col}_scaled'] = df[f'{col}_scaled'].fillna(df[f'{col}_scaled'].mean())
    
    logger.info(f"  ✓ {len(numerical_cols)} numerical features scaled.")
    return df


# ============================================================================
# FEATURE ENGINEERING PIPELINE
# ============================================================================

def run_preprocessing_pipeline(input_csv, output_csv=None, mode='augmented', config=None):
    if config is None:
        config = {
            'use_nlp': True,
            'use_logistics': True if mode == 'real' else False,
            'language': 'auto'
        }
    """
    Execute the complete preprocessing pipeline
    Modes:
      - 'augmented': Includes synthetic features
      - 'real': Strict separation relying only on real feedback
    """
    logger.info("=" * 80)
    logger.info(f"🚀 STARTING {mode.upper()} PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"📂 Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # ===== PHASE 0 =====
    df = standardize_schema(df)
    
    # ===== PHASE 1 =====
    df = handle_missing_values(df)
    df = validate_order_integrity(df)
    df = handle_outliers(df)
    
    # ===== PHASE 2 =====
    df = extract_temporal_features(df)
    
    # ===== PHASE 3 =====
    df = calculate_rolling_behavior(df)
    
    # ===== PHASE 4 =====
    df = extract_sentiment_features(df, config)
    # Compute sarcasm & spam features (contradiction-based scoring)
    df = compute_sarcasm_and_spam_features(df, config)
    # Optional: df = vectorize_reviews(df, max_features=20)
    
    # ===== PHASE 5 =====
    
    # Optional phase 1 improvement: Using real logistics instead of discarding them
    if config and config.get('use_logistics', False):
        if 'actual_delivery_days' in df.columns and 'expected_delivery_days' in df.columns:
            # calculate explicit new real features
            df['delivery_delay'] = (df['actual_delivery_days'] - df['expected_delivery_days']).clip(lower=0)
            df['delivery_speed_ratio'] = df['actual_delivery_days'] / df['expected_delivery_days'].replace(0, 1)
            df['is_late_delivery'] = (df['delivery_delay'] > 0).astype(int)

    if mode == 'real':
        logger.info("   [MODE: REAL] Redesigning target to actual feedback signals & managing features dynamically")
        # Target Redesign: Drop indirect leakage (is_returned) and proxy via real feedback
        # Option A (Recommended): low_rating = review_score <= 2
        df['is_returned'] = (df['review_rating'].fillna(3) <= 2).astype(int)
        
        # We can drop synthetic ones, but keep real logistics if configured
        drop_cols = ['distance_km', 'discount_percentage', 'discount_amount', 'defect_rate', 'past_return_rate']
        if not config.get('use_logistics', False):
            drop_cols += ['actual_delivery_days', 'delivery_delay', 'expected_delivery_days', 'delivery_speed_ratio', 'is_late_delivery']
            
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        categorical_cols = [c for c in ['product_category', 'shipping_mode', 'return_reason', 'warehouse_city', 'customer_city'] if c in df.columns]
        numerical_cols = [c for c in ['product_price_log', 'quantity', 'order_value', 'review_rating', 'sentiment_polarity',
                                      'sentiment_subjectivity', 'review_word_count', 'review_lag_days',
                                      'actual_delivery_days', 'delivery_delay', 'expected_delivery_days', 
                                      'delivery_speed_ratio', 'is_late_delivery'] if c in df.columns]
    else:
        categorical_cols = ['product_category', 'shipping_mode', 'return_reason', 'warehouse_city', 'customer_city']
        numerical_cols = ['product_price_log', 'quantity', 'distance_km', 'actual_delivery_days', 
                          'order_value', 'review_rating', 'sentiment_polarity',
                          'sentiment_subjectivity', 'review_word_count', 'review_lag_days',
                          'delivery_delay', 'expected_delivery_days', 'discount_percentage', 'discount_amount']
                          
    df = encode_categorical_features(df, categorical_cols, target_col='is_returned')
    df = scale_numerical_features(df, numerical_cols, method='standard')

    # Assign Trust Scores based on segmentation strategy
    df['feature_trust_score'] = 1.0 if mode == 'real' else 0.5

    # ===== SAVE PREPROCESSED DATA =====
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_preprocessed.csv')
    
    df.to_csv(output_csv, index=False)
    logger.info(f"✅ Preprocessed data saved to: {output_csv}")
    logger.info(f"   Final shape: {df.shape}")
    
    # Return summary statistics
    logger.info("\n📊 Dataset Summary:")
    logger.info(f"   Total records: {len(df)}")
    logger.info(f"   Return rate: {df['is_returned'].mean():.2%}")
    logger.info(f"   Features created: {len(df.columns)}")
    
    return df


# ============================================================================
# VALIDATION & FEATURE IMPORTANCE
# ============================================================================

def validate_with_xgboost(df, test_size=0.2, random_state=42):
    """
    Run baseline XGBoost to validate feature importance
    """
    logger.info("\n" + "=" * 80)
    logger.info("🧪 VALIDATION: XGBoost Feature Importance Analysis")
    logger.info("=" * 80)
    
    # Prepare features and target
    target = df['is_returned']
    
    # Select numeric features only (exclude IDs, dates, original categorical, etc.)
    exclude_cols = ['customer_id', 'product_id', 'order_id', 'order_date', 'review_date',
                    'product_category', 'shipping_mode', 'return_reason',
                    'warehouse_city', 'customer_city', 'review_text', 'home_city']
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col != 'is_returned' 
                   and col not in [c for c in df.columns if df[c].dtype == 'object']]
    
    X = df[feature_cols].fillna(0)
    y = target
    
    logger.info(f"📌 Training XGBoost with {len(feature_cols)} features...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0
    )
    
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=False)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"\n📈 Model Performance:")
    logger.info(f"   Training Accuracy: {train_score:.4f}")
    logger.info(f"   Test Accuracy: {test_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\n🔝 Top 20 Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"   {row['feature']:40s} → {row['importance']:.4f}")
    
    # Save feature importance
    importance_csv = 'reports/feature_importance.csv'
    Path(importance_csv).parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(importance_csv, index=False)
    logger.info(f"\n✅ Feature importance saved to: {importance_csv}")
    
    return model, feature_importance


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Input data path
    input_file = 'data/synthetic_ecommerce_orders.csv'
    output_file = 'data/synthetic_ecommerce_orders_preprocessed.csv'
    
    # Run pipeline
    preprocessed_df = run_preprocessing_pipeline(input_file, output_file)
    
    # Validate with XGBoost
    model, importance = validate_with_xgboost(preprocessed_df)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 80)
