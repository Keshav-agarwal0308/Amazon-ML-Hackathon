# scripts/run_text_on_sample.py
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_IN = os.path.join(DATA_DIR, "sample_test.csv")
SAMPLE_GT = os.path.join(DATA_DIR, "sample_test_out.csv")   # has ground-truth for 100 rows
SAMPLE_PRED = os.path.join(DATA_DIR, "sample_test_pred.csv")

MAX_FEATURES = 200_000  # reduce to 100_000 if your machine is slow

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.99)  # enforce positive
    return float(np.mean(np.abs(y_pred - y_true) /
                         ((np.abs(y_true) + np.abs(y_pred) + eps) / 2.0)))

print("[1/5] Load data...")
train = pd.read_csv(TRAIN_CSV)
samp  = pd.read_csv(SAMPLE_IN)
gt    = pd.read_csv(SAMPLE_GT)  # sample_id, price

# basic hygiene on train
train = train.dropna(subset=['catalog_content']).copy()
train = train[train['price'] > 0].reset_index(drop=True)

print("[2/5] TF-IDF vectorize (fit on full train, transform sample_test)...")
vec = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                      ngram_range=(1,2), min_df=5, max_features=MAX_FEATURES, sublinear_tf=True)
X_tr = vec.fit_transform(train['catalog_content'].fillna(''))
X_sa = vec.transform(samp['catalog_content'].fillna(''))

y = train['price'].values.astype(float)

print("[3/5] Train Ridge on full train...")
model = Ridge(alpha=5.0, random_state=42)  # good alpha from your earlier run
model.fit(X_tr, y)

print("[4/5] Predict on sample_test...")
pred = np.maximum(model.predict(X_sa), 0.99)

pred_df = pd.DataFrame({'sample_id': samp['sample_id'], 'price': pred})
pred_df.to_csv(SAMPLE_PRED, index=False)
print(f"Saved predictions: {SAMPLE_PRED}")

print("[5/5] Evaluate against provided sample_test_out.csv...")
merged = pred_df.merge(gt, on='sample_id', suffixes=('_pred', '_true'))
if 'price_true' not in merged.columns:
    # if merge used default names, fix them
    merged = merged.rename(columns={'price_x':'price_pred', 'price_y':'price_true'})
score = smape(merged['price_true'].values, merged['price_pred'].values)
print(f"SMAPE on sample_test: {score:.4f}")
print(merged.head())
