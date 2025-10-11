# Quick Text Baseline Training Script

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_text import train_text_models
from src.smape import smape

print("Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

print("Training text models...")
preds_text = train_text_models(train, test, folds=None, outdir='models')

print("Saving predictions...")
pred_df = pd.DataFrame({
    'sample_id': test['sample_id'],
    **{k: v for k,v in preds_text.items()}
})
pred_df.to_parquet('features/test_text_preds.parquet', index=False)

print("Creating text-only submission...")
# Create submission with best text model (Ridge)
submission = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': preds_text['Text_Ridge']
})
submission.to_csv('dataset/test_out_text_only.csv', index=False)

print("Text-only submission saved to dataset/test_out_text_only.csv")
print("Model predictions saved to features/test_text_preds.parquet")

# Quick validation
print(f"\nSubmission stats:")
print(f"Shape: {submission.shape}")
print(f"Price range: {submission['price'].min():.2f} - {submission['price'].max():.2f}")
print(f"Sample IDs unique: {submission['sample_id'].is_unique}")
print(f"No missing values: {submission.isnull().sum().sum() == 0}")
