import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import pickle
import logging
from pathlib import Path
import json
import os

# Create Insights Exporter Tool
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from insights_extractor import save_insights, extract_feature_importances

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path('data/synthetic_ecommerce_orders.csv')
MODELS_DIR = Path(__file__).parent.parent / 'models' / 'core_predictors'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not DATA_PATH.exists():
        logger.error(f"Dataset not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df

def build_pipeline(num_features, cat_features, is_classification=True, scale_pos_weight=1):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    if is_classification:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)
    else:
        model = XGBRegressor(random_state=42)
        
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def train_return_predictor(df):
    logger.info("--- Model A: Product Return Prediction -> `is_returned` ---")
    
    # Target: is_returned
    target = 'is_returned'
    
    # Exclude post-purchase leakage data (e.g. return_reason, review_text) for a clean, predictive setup
    num_features = ['discount_percentage', 'distance_km', 'past_return_rate', 'base_return_tendency', 'defect_rate', 'delivery_delay']
    cat_features = ['product_category', 'shipping_mode', 'is_remote_area']
    
    X = df[num_features + cat_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Class Imbalance
    class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    logger.info(f"Class ratio for scale_pos_weight: {class_ratio:.2f}")

    # --- Hyperparameter Tuning via Optuna ---
    try:
        import optuna
        logger.info("Starting Bayesian Optimization with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'random_state': 42,
                'scale_pos_weight': class_ratio
            }
            
            # Transformer
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])
            
            model = XGBClassifier(**params)
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Fit and evaluate
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            return f1_score(y_test, y_pred)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10) # 10 trials for speed
        
        best_params = study.best_params
        logger.info(f"Best hyperparameters found: {best_params}")
        
    except ImportError:
        logger.warning("Optuna not installed. Skipping hyperparameter tuning.")
        best_params = {}

    # Build final pipeline with best params
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])
    
    final_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=class_ratio, **best_params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Save Insights
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred))
    }
    ft_importances = extract_feature_importances(pipeline, num_features, cat_features, X_train)
    save_insights("Return Predictor", metrics, ft_importances)
    
    # Save Model
    with open(MODELS_DIR / 'return_predictor.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

def train_delivery_delay_forecaster(df):
    logger.info("--- Model B: Delivery Delay Prediction -> `delivery_delay` ---")
    
    target = 'delivery_delay'
    # Use features known at order creation time
    num_features = ['distance_km', 'expected_delivery_days']
    cat_features = ['warehouse_city', 'customer_city', 'shipping_mode', 'is_remote_area', 'product_category']
    
    X = df[num_features + cat_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        import optuna
        logger.info("Starting tuning for Delay Forecaster...")
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42
            }
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])
            model = XGBRegressor(**params)
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            return r2_score(y_test, preds)
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=15)
        best_params = study.best_params
    except ImportError:
        best_params = {}

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])
    
    final_model = XGBRegressor(random_state=42, **best_params) if best_params else XGBRegressor(random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} days")
    logger.info(f"R2  : {r2_score(y_test, y_pred):.4f}")
    
    # Save Insights
    metrics = {
        "rmse_days": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_score": float(r2_score(y_test, y_pred))
    }
    ft_importances = extract_feature_importances(pipeline, num_features, cat_features, X_train)
    save_insights("Delay Forecaster", metrics, ft_importances)

    with open(MODELS_DIR / 'delay_forecaster.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

def train_satisfaction_predictor(df):
    logger.info("--- Model C: Customer Satisfaction Prediction -> `review_rating` ---")
    
    # Only train on complete targets
    df_clean = df.dropna(subset=['review_rating']).copy()
    target = 'review_rating'
    
    num_features = ['delivery_delay', 'defect_rate', 'discount_percentage', 'past_return_rate']
    cat_features = ['product_category', 'is_returned']  # Since this is a post-purchase feedback estimator
    
    X = df_clean[num_features + cat_features]
    y = df_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline(num_features, cat_features, is_classification=False)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} stars")
    logger.info(f"R2  : {r2_score(y_test, y_pred):.4f}")

    # Save Insights
    metrics = {
        "rmse_stars": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_score": float(r2_score(y_test, y_pred))
    }
    ft_importances = extract_feature_importances(pipeline, num_features, cat_features, X_train)
    save_insights("CSAT Predictor", metrics, ft_importances)

    with open(MODELS_DIR / 'csat_predictor.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

def train_revenue_predictor(df):
    logger.info("--- Model D: Revenue and Order Value -> `order_value` ---")
    
    target = 'order_value'
    num_features = ['quantity', 'product_price', 'discount_percentage', 'past_return_rate']
    cat_features = ['product_category', 'customer_city']
    
    X = df[num_features + cat_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline(num_features, cat_features, is_classification=False)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    logger.info(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    logger.info(f"R2  : {r2_score(y_test, y_pred):.4f}")

    # Save Insights
    metrics = {
        "rmse_dollars": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2_score": float(r2_score(y_test, y_pred))
    }
    ft_importances = extract_feature_importances(pipeline, num_features, cat_features, X_train)
    save_insights("Revenue Predictor", metrics, ft_importances)

    with open(MODELS_DIR / 'revenue_predictor.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        train_return_predictor(df)
        train_delivery_delay_forecaster(df)
        train_satisfaction_predictor(df)
        train_revenue_predictor(df)
        logger.info("✅ All core ML models successfully trained and exported!")
