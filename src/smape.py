import numpy as np

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # enforce positivity to be safe
    y_pred = np.maximum(y_pred, 0.99)
    num = np.abs(y_pred - y_true)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(num / denom))
