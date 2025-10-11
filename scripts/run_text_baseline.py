# scripts/run_text_baseline.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

# ---------- CONFIG ----------
DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
SUBMIT_CSV = os.path.join(DATA_DIR, "test_out.csv")

N_FOLDS = 5
MAX_FEATURES = 200_000   # adjust if RAM is tight (e.g., 150k)

# ---------- Metric: SMAPE ----------
def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.maximum(y_pred, 0.99)  # enforce positive for stability
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(num / denom))

print("Loading data...")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# Basic hygiene
train = train.dropna(subset=['catalog_content']).copy()
train = train[train['price'] > 0].reset_index(drop=True)
test  = test.copy()

# ---------- TF-IDF features (text only baseline) ----------
print("Vectorizing text with TF-IDF...")
vec_word = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode',
    ngram_range=(1,2),      # word unigrams + bigrams
    min_df=5,
    max_features=MAX_FEATURES,
    sublinear_tf=True
)
X_tr = vec_word.fit_transform(train['catalog_content'].fillna(''))
X_te = vec_word.transform(test['catalog_content'].fillna(''))

y = train['price'].values.astype(float)

# ---------- Cross-validation to estimate SMAPE ----------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
alphas = [0.5, 1.0, 2.0, 5.0, 10.0]

print(f"Running {N_FOLDS}-fold CV over Ridge alphas {alphas} ...")
results = []
best_alpha = None
best_cv = 1e9

for alpha in alphas:
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr), 1):
        Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        model = Ridge(alpha=alpha, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        pred = np.maximum(pred, 0.99)
        score = smape(yva, pred)
        fold_scores.append(score)
        print(f"  alpha={alpha} fold={fold} SMAPE={score:.4f}")

    cv_mean = float(np.mean(fold_scores))
    results.append((alpha, cv_mean))
    print(f"alpha={alpha} CV SMAPE: {cv_mean:.4f}")
    if cv_mean < best_cv:
        best_cv = cv_mean
        best_alpha = alpha

print("\nCV results:")
for a, s in results:
    print(f"  alpha={a}: SMAPE={s:.4f}")
print(f"\nBest alpha={best_alpha} with CV SMAPE={best_cv:.4f}")

# ---------- Retrain on FULL train with best alpha ----------
print(f"\nTraining final Ridge on all {len(train)} rows (alpha={best_alpha}) ...")
final_model = Ridge(alpha=best_alpha, random_state=42)
final_model.fit(X_tr, y)

# ---------- Predict test and write submission ----------
print("Predicting on test ...")
test_pred = final_model.predict(X_te)
test_pred = np.maximum(test_pred, 0.99).astype('float32')

submit = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': test_pred
})

# Sanity checks
assert submit['price'].min() > 0, "Predicted non-positive price!"
assert len(submit) == len(test), "Mismatch in row count!"
assert submit['sample_id'].is_unique, "Duplicate sample_id!"

submit.to_csv(SUBMIT_CSV, index=False)
print(f"\nSaved submission to: {SUBMIT_CSV}")
print(submit.head())
