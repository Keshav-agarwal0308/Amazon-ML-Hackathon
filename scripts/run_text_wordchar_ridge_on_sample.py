import os, numpy as np, pandas as pd, re, time
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_IN = os.path.join(DATA_DIR, "sample_test.csv")
SAMPLE_GT = os.path.join(DATA_DIR, "sample_test_out.csv")
OUT_PRED  = os.path.join(DATA_DIR, "sample_test_pred_ridge.csv")

MAX_WORD_FEATS = 120_000   # smaller -> faster
MAX_CHAR_FEATS = 60_000

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

print("[1/5] Load")
train = pd.read_csv(TRAIN_CSV)
samp  = pd.read_csv(SAMPLE_IN)
gt    = pd.read_csv(SAMPLE_GT)
train = train.dropna(subset=['catalog_content'])
train = train[train['price']>0].reset_index(drop=True)

print("[2/5] TF-IDF (word+char) + numeric")
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

def make_num(df):
    txt = df['catalog_content'].astype(str)
    qty = txt.apply(extract_quantity).astype('float32')
    len_chars = txt.apply(len).astype('float32')
    len_words = txt.apply(lambda s: len(s.split())).astype('float32')
    digit_ratio = txt.apply(lambda s: sum(ch.isdigit() for ch in s)/max(1,len(s))).astype('float32')
    return csr_matrix(np.vstack([qty,len_chars,len_words,digit_ratio]).T)

Xn_tr = make_num(train)
Xn_sa = make_num(samp)

from scipy.sparse import hstack
X_tr = hstack([Xw_tr, Xc_tr, Xn_tr]).tocsr()
X_sa = hstack([Xw_sa, Xc_sa, Xn_sa]).tocsr()
y = train['price'].values.astype(float)

print("[3/5] Train Ridge (alpha=5.0)")
model = Ridge(alpha=5.0, random_state=42)
t0=time.time(); model.fit(X_tr, y); print(f"  fit in {time.time()-t0:.1f}s")

print("[4/5] Predict & save")
pred = np.maximum(model.predict(X_sa), 0.99)
pred = pred.astype('float32')
pd.DataFrame({'sample_id':samp['sample_id'], 'price':pred}).to_csv(OUT_PRED, index=False)
print("Saved:", OUT_PRED)

print("[5/5] Evaluate vs sample_test_out.csv")
m = pd.read_csv(OUT_PRED).merge(gt, on='sample_id', suffixes=('_pred','_true'))
if 'price_true' not in m.columns:
    m = m.rename(columns={'price_x':'price_pred','price_y':'price_true'})
print("SMAPE:", smape(m['price_true'], m['price_pred']))
print(m.head())
