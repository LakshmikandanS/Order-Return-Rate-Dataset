import os
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add scripts directory to path to allow importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocessing'))

from scripts.train_core_predictors import (
    train_return_predictor,
    train_delivery_delay_forecaster,
    train_satisfaction_predictor,
    train_revenue_predictor
)

def run_pipeline(input_csv_path):
    logger.info("====================================================")
    logger.info(f"🚀 Starting Unified ML Pipeline for: {input_csv_path}")
    logger.info("====================================================\n")

    input_path = Path(input_csv_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    logger.info("[1/2] Loading Data...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return
        
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Train Core ML Predictors & Extract Insights
    # ---------------------------------------------------------
    logger.info("\n[2/2] Training Core Predictors & Extracting Automated Insights...")
    
    # We define the directory relative to the pipeline
    from scripts.train_core_predictors import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_return_predictor(df)
    train_delivery_delay_forecaster(df)
    train_satisfaction_predictor(df)
    train_revenue_predictor(df)

    logger.info("\n====================================================")
    logger.info("✅ Unified Pipeline Execution completed successfully!")
    logger.info("Models are saved in models/core_predictors")
    logger.info("Automated ML Insights are extracted to insights")
    logger.info("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Machine Learning Pipeline")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="data/synthetic_ecommerce_orders.csv",
        help="Path to the input CSV dataset"
    )
    args = parser.parse_args()

    # Move working directory to the script's root folder to ensure relative paths work
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    run_pipeline(args.dataset)
