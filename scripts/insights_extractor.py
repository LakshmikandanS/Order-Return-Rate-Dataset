import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

INSIGHTS_DIR = Path(__file__).parent.parent / 'insights'
INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def save_insights(model_name, metrics, feature_importances=None):
    """
    Save the extracted metrics and insights to the insights directory.
    """
    insight_data = {
        'model_name': model_name,
        'metrics': metrics
    }
    
    if feature_importances:
        insight_data['top_features'] = feature_importances
        
    out_file = INSIGHTS_DIR / f"{model_name.lower().replace(' ', '_')}_insights.json"
    
    with open(out_file, 'w') as f:
        json.dump(insight_data, f, indent=4)
        
    logger.info(f"Saved insights for {model_name} to {out_file}")

def extract_feature_importances(pipeline, num_features, cat_features, X_train=None):
    """
    Helper function to extract XGBoost feature importances from the pipeline
    If X_train is provided, computes SHAP values for deeper explainability.
    """
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        # Get categorical feature names after OneHotEncoding
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()

        all_features = num_features + cat_feature_names
        
        # Determine Explainability Method
        importances = None
        if X_train is not None:
            try:
                import shap
                import numpy as np
                logger.info("Computing SHAP values for deeper explainability...")
                X_transformed = preprocessor.transform(X_train)
                # Ensure dense array for SHAP
                X_transformed_dense = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed_dense)
                
                # Global feature importance is the mean absolute SHAP value
                importances = np.abs(shap_values).mean(axis=0)
                logger.info("SHAP values calculated successfully.")
            except ImportError:
                logger.warning("SHAP not installed. Falling back to default XGBoost feature importances.")
                importances = model.feature_importances_
        else:
            importances = model.feature_importances_
        
        # Sort features by importance
        feature_importance_dict = dict(zip(all_features, importances))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
        
        # Return top 10
        return [{"feature": f, "importance": float(imp)} for f, imp in sorted_features[:10]]
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return None
