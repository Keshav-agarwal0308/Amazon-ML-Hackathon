import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA

def train_image_model(train_df, test_df, feat_prefix='features/img', out_path='models/img_lgbm.joblib'):
    Xtr = np.load(f'{feat_prefix}_train.npy')
    Xte = np.load(f'{feat_prefix}_test.npy')
    mtr = np.load(f'{feat_prefix}_train_missing.npy')
    mte = np.load(f'{feat_prefix}_test_missing.npy')

    # simple stats
    Xtr2 = np.concatenate([Xtr, mtr[:,None]], axis=1)
    Xte2 = np.concatenate([Xte, mte[:,None]], axis=1)

    # optional PCA to 256
    pca = PCA(n_components=256, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)

    Ztr = np.concatenate([Ztr, mtr[:,None]], axis=1)
    Zte = np.concatenate([Zte, mte[:,None]], axis=1)

    y = train_df['price'].values.astype(float)

    ds = lgb.Dataset(Ztr, label=y, free_raw_data=False)
    params = dict(
        objective='mae',
        learning_rate=0.07,
        num_leaves=128,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=60,
        seed=42,
        verbose=-1,
        n_jobs=-1,
    )
    model = lgb.train(params, ds, num_boost_round=1500)
    pred = np.maximum(model.predict(Zte), 0.99)

    joblib.dump({'model': model, 'pca': pca}, out_path)
    return pred
