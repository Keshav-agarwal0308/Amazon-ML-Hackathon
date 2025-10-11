import numpy as np
import pandas as pd

def non_negative_grid_search(oof_dict, y, weight_grid=None):
    # oof_dict: name -> array (len=n_train)  [You can extend to OOF later]
    # Here we don't have OOF for all models in this minimal path,
    # so we use a simple fixed grid on test only. When you have OOF,
    # compute SMAPE on OOF to pick weights.
    if weight_grid is None:
        weight_grid = [
            {'Text_Ridge':0.5, 'Text_ElasticNet':0.3, 'Text_LGBM':0.1, 'Image_LGBM':0.1},
            {'Text_Ridge':0.35, 'Text_ElasticNet':0.15,'Text_LGBM':0.35,'Image_LGBM':0.15},
            {'Text_Ridge':0.4, 'Text_ElasticNet':0.2, 'Text_LGBM':0.3, 'Image_LGBM':0.1},
            {'Text_Ridge':0.5, 'Text_ElasticNet':0.0, 'Text_LGBM':0.4, 'Image_LGBM':0.1},
            {'Text_Ridge':0.45,'Text_ElasticNet':0.15,'Text_LGBM':0.25,'Image_LGBM':0.15},
        ]
    # If no OOF, just return the first as default
    return weight_grid[1]

def apply_blend(test_df, text_preds_parquet, image_preds_parquet, out_csv):
    t = pd.read_parquet(text_preds_parquet)
    i = pd.read_parquet(image_preds_parquet)
    df = t.merge(i, on='sample_id', how='left')

    # fill missing columns if any model absent
    for k in ['Text_Ridge','Text_ElasticNet','Text_LGBM','Image_LGBM']:
        if k not in df.columns:
            df[k] = df.filter(like=k.split('_')[0]).iloc[:,0] if any(df.columns.str.contains(k.split('_')[0])) else 0.0

    # choose weights (static here; if you have OOF, compute best on OOF CV)
    w = non_negative_grid_search({}, None)

    pred = (w['Text_Ridge']     * df['Text_Ridge'] +
            w['Text_ElasticNet']* df['Text_ElasticNet'] +
            w['Text_LGBM']      * df['Text_LGBM'] +
            w['Image_LGBM']     * df['Image_LGBM']).values

    # positivity + tail clipping
    pred = np.maximum(pred, 0.99)

    # Optional distribution clipping: use train quantiles if you load them here.
    out = pd.DataFrame({'sample_id': df['sample_id'], 'price': pred})
    out.to_csv(out_csv, index=False)
    return out_csv
