import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train_sarcasm_spam_models(data_path):
    logger.info(f"Loading preprocessed data from {data_path}")
    df = pd.read_csv(data_path)
    
    # We need labels for Sarcasm and Spam to train an ML model.
    # Since we don't have ground truth labels natively, we will use our high-confidence heuristics 
    # from the spam_sarcastic_handling_strategy to create pseudo-labels for training, 
    # or expose the heuristic scores directly to a downstream ML algorithm.
    
    if 'is_sarcastic_score_flag' not in df.columns or 'is_likely_spam' not in df.columns:
        logger.error("Heuristic flags not found. Please ensure preprocess.py has run completely.")
        return
        
    logger.info("Preparing data for ML Classification (TF-IDF + Heuristic Features) over pseudo-labels...")
    
    # Text Feature Extraction
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    df['review_text'] = df['review_text'].fillna('')
    X_text = tfidf.fit_transform(df['review_text']).toarray()
    
    # ML for Sarcasm
    logger.info("\n--- Training Sarcasm ML Classifier ---")
    # Target: is_sarcastic_score_flag (from our strategy)
    y_sarcasm = df['is_sarcastic_score_flag']
    
    # Combine text features with ML context
    X_sarcasm_numeric = df[['sentiment_polarity', 'actual_delivery_days', 'expected_delivery_days', 'review_rating', 'defect_rate']].fillna(0).values
    X_sarcasm = np.hstack((X_text, X_sarcasm_numeric))
    
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_sarcasm, y_sarcasm, test_size=0.2, random_state=42)
    sarcasm_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    sarcasm_model.fit(Xs_train, ys_train)
    
    ys_pred = sarcasm_model.predict(Xs_test)
    logger.info(f"Sarcasm Model Accuracy: {accuracy_score(ys_test, ys_pred):.4f}")
    logger.info("Sarcasm Classification Report:\n" + classification_report(ys_test, ys_pred))
    
    # ML for Spam
    logger.info("\n--- Training Spam ML Classifier ---")
    y_spam = df['is_likely_spam']
    
    X_spam_numeric = df[['review_word_count', 'sentiment_polarity', 'review_rating', 'past_return_rate', 'total_orders']].fillna(0).values
    X_spam = np.hstack((X_text, X_spam_numeric))
    
    Xsp_train, Xsp_test, ysp_train, ysp_test = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42)
    spam_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    spam_model.fit(Xsp_train, ysp_train)
    
    ysp_pred = spam_model.predict(Xsp_test)
    logger.info(f"Spam Model Accuracy: {accuracy_score(ysp_test, ysp_pred):.4f}")
    logger.info("Spam Classification Report:\n" + classification_report(ysp_test, ysp_pred))
    
    # Save the models
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / 'sarcasm_classifier.pkl', 'wb') as f:
        pickle.dump(sarcasm_model, f)
    with open(models_dir / 'spam_classifier.pkl', 'wb') as f:
        pickle.dump(spam_model, f)
    with open(models_dir / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    logger.info("✅ Models saved to models/ directory successfully.")

if __name__ == "__main__":
    data_file = 'data/synthetic_ecommerce_orders_preprocessed.csv'
    train_sarcasm_spam_models(data_file)
