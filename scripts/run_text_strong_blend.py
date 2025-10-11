import sys, faulthandler; faulthandler.enable(); sys.tracebacklimit = 100
import os, re, time, numpy as np, pandas as pd
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb

DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
OUT_CSV   = os.path.join(DATA_DIR, "test_out.csv")

# ---------------- SMAPE (with formula in docstring)
def smape(y_true, y_pred, eps=1e-6):
    """
    SMAPE = (1/n) * sum( |p - a| / ((|a| + |p|)/2) )
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_pred = np.maximum(y_pred, 0.99)
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(num / denom))

# --------------- lightweight numeric features that matter for price
def extract_quantity(text):
    t = str(text).lower()
    for p in [r'pack\s*of\s*(\d+)', r'(\d+)\s*(?:pcs|pieces|count)\b', r'\bx\s*(\d+)\b', r'(\d+)\s*x\b']:
        m = re.search(p, t)
        if m:
            try: return max(1, int(m.group(1)))
            except: pass
    return 1

def make_numeric(df):
    txt = df['catalog_content'].astype(str)
    qty = txt.apply(extract_quantity).astype('float32')
    len_chars = txt.apply(len).astype('float32')
    len_words = txt.apply(lambda s: len(s.split())).astype('float32')
    digit_ratio = txt.apply(lambda s: sum(ch.isdigit() for ch in s)/max(1,len(s))).astype('float32')
    return csr_matrix(np.vstack([qty, len_chars, len_words, digit_ratio]).T), ['qty','len_chars','len_words','digit_ratio']

# ---------------- Load
print("[1/8] Load train/test")
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

train = train.dropna(subset=['catalog_content']).copy()
train = train[train['price'] > 0].reset_index(drop=True)

# ---------------- TF-IDF (word + char)
print("[2/8] Build TF-IDF (word 1–2, char 3–5)")
MAX_WORD = 150_000
MAX_CHAR = 100_000
vec_w = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                        ngram_range=(1,2), min_df=5, max_features=MAX_WORD, sublinear_tf=True)
vec_c = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                        analyzer='char', ngram_range=(3,5), min_df=5, max_features=MAX_CHAR, sublinear_tf=True)

Xw_tr = vec_w.fit_transform(train['catalog_content'].fillna(''))
Xw_te = vec_w.transform(test['catalog_content'].fillna(''))
Xc_tr = vec_c.fit_transform(train['catalog_content'].fillna(''))
Xc_te = vec_c.transform(test['catalog_content'].fillna(''))

# numeric
Xn_tr, num_cols = make_numeric(train)
Xn_te, _        = make_numeric(test)

# sparse block for Ridge
from scipy.sparse import hstack
X_tr = hstack([Xw_tr, Xc_tr, Xn_tr]).tocsr()
X_te = hstack([Xw_te, Xc_te, Xn_te]).tocsr()

y = train['price'].values.astype('float32')

# ---------------- Ridge with OOF (for blending)
print("[3/8] Ridge OOF + model")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_r = np.zeros(len(train), dtype='float32')
ridge = Ridge(alpha=5.0, random_state=42)

for fold, (tr, va) in enumerate(kf.split(X_tr), 1):
    m = Ridge(alpha=5.0, random_state=42)
    m.fit(X_tr[tr], y[tr])
    p = m.predict(X_tr[va]).astype('float32')
    oof_r[va] = np.maximum(p, 0.99)
    print(f"  Ridge fold {fold} SMAPE={smape(y[va], oof_r[va]):.4f}")

ridge.fit(X_tr, y)  # final fit on all data
pred_r_test = np.maximum(ridge.predict(X_te).astype('float32'), 0.99)

print(f"  Ridge OOF SMAPE={smape(y, oof_r):.4f}")

# ---------------- SVD + LightGBM (captures nonlinearity)
print("[4/8] SVD(256) on word+char TF-IDF for LGBM")
svd = TruncatedSVD(n_components=256, random_state=42)
Z_tr = svd.fit_transform(hstack([Xw_tr, Xc_tr]))
Z_te = svd.transform(hstack([Xw_te, Xc_te]))

# concat numeric
Z_tr = np.hstack([Z_tr, np.asarray(Xn_tr.todense())])
Z_te = np.hstack([Z_te, np.asarray(Xn_te.todense())])

print("[5/8] LGBM OOF + model")
params = dict(
    objective='mae',   # proxy; we evaluate SMAPE separately
    learning_rate=0.07,
    num_leaves=128,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=80,
    seed=42,
    verbose=-1,
    n_jobs=-1
)
oof_l = np.zeros(len(train), dtype='float32')
for fold, (tr, va) in enumerate(kf.split(Z_tr), 1):
    dtr = lgb.Dataset(Z_tr[tr], label=y[tr], free_raw_data=False)
    dva = lgb.Dataset(Z_tr[va], label=y[va], free_raw_data=False)
    model = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], valid_names=['val'],
                      callbacks=[lgb.log_evaluation(period=0)])
    p = model.predict(Z_tr[va]).astype('float32')
    oof_l[va] = np.maximum(p, 0.99)
    print(f"  LGBM fold {fold} SMAPE={smape(y[va], oof_l[va]):.4f}")
# final fit
dall = lgb.Dataset(Z_tr, label=y, free_raw_data=False)
lgbm = lgb.train(params, dall, num_boost_round=1500)
pred_l_test = np.maximum(lgbm.predict(Z_te).astype('float32'), 0.99)

print(f"  LGBM OOF SMAPE={smape(y, oof_l):.4f}")

# ---------------- Blend weights via OOF SMAPE grid search
print("[6/8] Blend weights via OOF SMAPE")
weights = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
best = None; best_s = 9e9
for w in weights:
    blend_oof = w*oof_r + (1-w)*oof_l
    s = smape(y, blend_oof)
    if s < best_s:
        best_s = s; best = w
print(f"  Best blend weight (Ridge) = {best:.2f} with OOF SMAPE={best_s:.4f}")

pred_test = best*pred_r_test + (1-best)*pred_l_test

# ---------------- Post-process: positive + tail clip (helps SMAPE)
print("[7/8] Post-process predictions")
p1, p995 = np.percentile(y, [1, 99.5])
pred_test = np.maximum(pred_test, 0.99)
pred_test = np.clip(pred_test, p1, p995)

# ---------------- Save
print("[8/8] Save submission")
sub = pd.DataFrame({'sample_id': test['sample_id'], 'price': pred_test.astype('float32')})
assert len(sub)==len(test) and sub['sample_id'].is_unique and sub['price'].min()>0
sub.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print(sub.head())
