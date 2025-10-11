import hashlib
import pandas as pd
from sklearn.model_selection import GroupKFold

def _normalize_title(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # keep letters/numbers, replace digits with #
    out = []
    for ch in t:
        if ch.isdigit():
            out.append('#')
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(' ')
    return ' '.join(''.join(out).split())

def title_hash(s: str) -> str:
    norm = _normalize_title(s)
    return hashlib.md5(norm.encode('utf-8')).hexdigest()[:10]

def make_group_folds(df: pd.DataFrame, n_splits=5, seed=42):
    groups = df['catalog_content'].fillna("").map(title_hash)
    gkf = GroupKFold(n_splits=n_splits)
    fold = -1 * pd.Series([1]*len(df), index=df.index, dtype=int)
    for k, (_, val_idx) in enumerate(gkf.split(df, groups=groups)):
        fold.iloc[val_idx] = k
    assert (fold >= 0).all()
    return fold
