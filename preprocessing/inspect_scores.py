import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('data/synthetic_ecommerce_orders_preprocessed.csv')
    cols = ['order_id','review_text','review_rating','sentiment_polarity','is_sarcastic','sarcasm_score','is_sarcastic_score_flag','spam_score','is_likely_spam']
    cols = [c for c in cols if c in df.columns]

    print('\nTop 10 by sarcasm_score:')
    top_sarcasm = df.sort_values('sarcasm_score', ascending=False).head(10)
    print(top_sarcasm[cols].to_string(index=False))

    print('\nTop 20 by spam_score:')
    top_spam = df.sort_values('spam_score', ascending=False).head(20)
    print(top_spam[cols].to_string(index=False))
