import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import vstack, hstack
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

from .smape import smape
from .fe_text import build_tfidf, make_numeric_df

def _onehot_brand(train_brand, test_brand, min_freq=20):
    # reduce cardinality
    vc = train_brand.value_counts()
    keep = set(vc[vc>=min_freq].index.tolist())
    tr = train_brand.where(train_brand.isin(keep), 'other').astype(str).values.reshape(-1,1)
    te = test_brand.where(test_brand.isin(keep), 'other').astype(str).values.reshape(-1,1)
    
    # ✅ Compatibility with all sklearn versions
    try:
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        # older sklearn uses `sparse` instead of `sparse_output`
        enc = OneHotEncoder(handle_unknown='ignore', sparse=True)
    
    Xtr = enc.fit_transform(tr)
    Xte = enc.transform(te)
    return Xtr, Xte, enc

def train_text_models(train_df, test_df, folds, outdir='models'):
    os.makedirs(outdir, exist_ok=True)

    # 1) vectorize text
    Xtr_tfidf, Xte_tfidf, wv, cv = build_tfidf(train_df['catalog_content'].fillna(""),
                                               test_df['catalog_content'].fillna(""))

    # 2) numeric features
    num_tr = make_numeric_df(train_df)
    num_te = make_numeric_df(test_df)

    # 3) brand one-hot
    Xb_tr, Xb_te, enc = _onehot_brand(num_tr['brand_norm'], num_te['brand_norm'])

    # 4) final matrices
    from scipy.sparse import csr_matrix
    numeric_cols = [c for c in num_tr.columns if c!='brand_norm']
    Xnum_tr = csr_matrix(num_tr[numeric_cols].values)
    Xnum_te = csr_matrix(num_te[numeric_cols].values)

    Xtr = hstack([Xtr_tfidf, Xb_tr, Xnum_tr]).tocsr()
    Xte = hstack([Xte_tfidf, Xb_te, Xnum_te]).tocsr()

    y = train_df['price'].values.astype(float)

    oof = {}
    preds = {}

    # ---- Ridge ----
    ridge = Ridge(alpha=2.0, random_state=42)
    oof_r = np.zeros(len(train_df))
    ridge.fit(Xtr, y)
    pred_r = np.maximum(ridge.predict(Xte), 0.99)
    oof['Text_Ridge'] = oof_r  # (optionally fill folds later)
    preds['Text_Ridge'] = pred_r
    joblib.dump({'model': ridge, 'wv': wv, 'cv': cv, 'enc': enc, 'numeric_cols': numeric_cols},
                os.path.join(outdir,'text_ridge.joblib'))

    # ---- ElasticNet ----
    enet = ElasticNet(alpha=0.01, l1_ratio=0.15, random_state=42, max_iter=2000)
    enet.fit(Xtr, y)
    pred_e = np.maximum(enet.predict(Xte), 0.99)
    preds['Text_ElasticNet'] = pred_e
    joblib.dump({'model': enet, 'wv': wv, 'cv': cv, 'enc': enc, 'numeric_cols': numeric_cols},
                os.path.join(outdir,'text_enet.joblib'))

    # ---- LightGBM ----
    # Use only numeric + brand one-hot for LGBM (avoids huge sparse)
    # Use SVD of TF-IDF for dimensionality reduction
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=128, random_state=42)  # Reduced from 256
    Xtr_svd = svd.fit_transform(Xtr_tfidf)
    Xte_svd = svd.transform(Xte_tfidf)

    import scipy.sparse as sp
    Xlgb_tr = sp.hstack([sp.csr_matrix(Xtr_svd), Xb_tr, Xnum_tr]).tocsr()
    Xlgb_te = sp.hstack([sp.csr_matrix(Xte_svd), Xb_te, Xnum_te]).tocsr()

    lgb_trn = lgb.Dataset(Xlgb_tr, label=y, free_raw_data=False)
    params = dict(
        objective='mae',  # proxy for SMAPE
        learning_rate=0.1,  # Increased for faster convergence
        num_leaves=64,  # Reduced for memory efficiency
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=100,  # Increased for stability
        verbose=-1,
        seed=42,
        n_jobs=1  # Reduced to avoid memory issues
    )
    model_lgb = lgb.train(params, lgb_trn, num_boost_round=800)  # Reduced iterations
    pred_l = np.maximum(model_lgb.predict(Xlgb_te), 0.99)
    preds['Text_LGBM'] = pred_l
    joblib.dump({'model': model_lgb, 'svd': svd, 'enc': enc, 'numeric_cols': numeric_cols},
                os.path.join(outdir,'text_lgbm.joblib'))

    # Return predictions for blending
    return preds
