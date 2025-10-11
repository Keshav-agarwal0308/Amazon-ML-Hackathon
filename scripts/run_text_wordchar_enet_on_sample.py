# scripts/run_text_wordchar_enet_on_sample.py
import os, numpy as np, pandas as pd, re
from tqdm import tqdm
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold

DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_IN = os.path.join(DATA_DIR, "sample_test.csv")
SAMPLE_GT = os.path.join(DATA_DIR, "sample_test_out.csv")
OUT_PRED  = os.path.join(DATA_DIR, "sample_test_pred_enet.csv")

MAX_WORD_FEATS = 150_000
MAX_CHAR_FEATS = 100_000
N_FOLDS = 5

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float)
    y_pred = np.maximum(np.asarray(y_pred, float), 0.99)
    return float(np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred) + eps)/2.0)))

def extract_quantity(text):
    t = str(text).lower()
    for p in [r'pack\s*of\s*(\d+)', r'(\d+)\s*(?:pcs|pieces|count)\b', r'\bx\s*(\d+)\b', r'(\d+)\s*x\b']:
        m = re.search(p, t)
        if m:
            try: return max(1, int(m.group(1)))
            except: pass
    return 1

print("[1/6] Load data")
train = pd.read_csv(TRAIN_CSV)
samp  = pd.read_csv(SAMPLE_IN)
gt    = pd.read_csv(SAMPLE_GT)

train = train.dropna(subset=['catalog_content']).copy()
train = train[train['price'] > 0].reset_index(drop=True)

print("[2/6] Build features (word + char TF-IDF + numeric)")
tr_text = train['catalog_content'].fillna('')
sa_text = samp['catalog_content'].fillna('')

vec_w = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                        ngram_range=(1,2), min_df=5, max_features=MAX_WORD_FEATS, sublinear_tf=True)
vec_c = TfidfVectorizer(lowercase=True, strip_accents='unicode',
                        analyzer='char', ngram_range=(3,5), min_df=5, max_features=MAX_CHAR_FEATS, sublinear_tf=True)

Xw_tr = vec_w.fit_transform(tr_text)
Xw_sa = vec_w.transform(sa_text)
Xc_tr = vec_c.fit_transform(tr_text)
Xc_sa = vec_c.transform(sa_text)

# simple numeric features
def make_num(df):
    txt = df['catalog_content'].astype(str)
    qty = txt.apply(extract_quantity).astype('float32')
    len_chars = txt.apply(len).astype('float32')
    len_words = txt.apply(lambda s: len(s.split())).astype('float32')
    digit_ratio = txt.apply(lambda s: sum(ch.isdigit() for ch in s)/max(1,len(s))).astype('float32')
    num = np.vstack([qty, len_chars, len_words, digit_ratio]).T
    return csr_matrix(num)

Xn_tr = make_num(train)
Xn_sa = make_num(samp)

from scipy.sparse import vstack
X_tr = hstack([Xw_tr, Xc_tr, Xn_tr]).tocsr()
X_sa = hstack([Xw_sa, Xc_sa, Xn_sa]).tocsr()

y = train['price'].values.astype(float)

print(f"Shapes: X_tr={X_tr.shape}, X_sa={X_sa.shape}")

print("[3/6] CV to tune ElasticNet")
grid = [(0.01,0.15),(0.02,0.15),(0.01,0.25),(0.02,0.25)]
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
best, bestcv = None, 9e9
for a,l1 in grid:
    scores=[]
    for tr,va in kf.split(X_tr):
        m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=4000, random_state=42)
        m.fit(X_tr[tr], y[tr])
        p = m.predict(X_tr[va]); p = np.maximum(p, 0.99)
        scores.append(smape(y[va], p))
    cv = float(np.mean(scores))
    print(f"  ENet alpha={a} l1={l1} CV SMAPE={cv:.4f}")
    if cv < bestcv: bestcv, best = cv, (a,l1)

print(f"[4/6] Best ENet: alpha={best[0]} l1={best[1]} CV SMAPE={bestcv:.4f}")

print("[5/6] Train final ENet on full train and predict sample")
enet = ElasticNet(alpha=best[0], l1_ratio=best[1], max_iter=4000, random_state=42)
enet.fit(X_tr, y)
pred_enet = np.maximum(enet.predict(X_sa), 0.99)

# Optional: light tail clipping to reduce huge errors
p1, p995 = np.percentile(y, [1, 99.5])
pred_enet = np.clip(pred_enet, p1, p995)

pred_df = pd.DataFrame({'sample_id': samp['sample_id'], 'price': pred_enet})
pred_df.to_csv(OUT_PRED, index=False)
print("Saved:", OUT_PRED)

print("[6/6] Evaluate vs sample_test_out.csv")
merged = pred_df.merge(gt, on='sample_id', suffixes=('_pred','_true'))
if 'price_true' not in merged.columns:
    merged = merged.rename(columns={'price_x':'price_pred','price_y':'price_true'})
score = smape(merged['price_true'].values, merged['price'].values)
print(f"SMAPE (ENet, word+char+num) on sample_test: {score:.4f}")
print(merged.head())
