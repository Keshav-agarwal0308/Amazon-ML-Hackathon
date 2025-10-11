import re
import regex as re2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- helpers ----------
_BRAND_RE = re.compile(r'^\s*([A-Z][\w&\-\']{1,15}(?:\s+[A-Z][\w&\-\']{1,15}){0,2})\b')
_UNITS = [
    ('ml', 1.0, r'(\d+(?:\.\d+)?)\s*ml\b'),
    ('l', 1000.0, r'(\d+(?:\.\d+)?)\s*l\b'),
    ('g', 1.0, r'(\d+(?:\.\d+)?)\s*g\b'),
    ('kg', 1000.0, r'(\d+(?:\.\d+)?)\s*kg\b'),
    ('cm', 1.0, r'(\d+(?:\.\d+)?)\s*cm\b'),
    ('mm', 0.1, r'(\d+(?:\.\d+)?)\s*mm\b'),
    ('inch', 2.54, r'(\d+(?:\.\d+)?)\s*(?:inch|in\.)\b'),
]
IPQ_PATTERNS = [
    r'pack\s*of\s*(\d+)',
    r'(\d+)\s*pcs\b',
    r'(\d+)\s*pieces\b',
    r'\b(\d+)\s*count\b',
    r'\bipq\s*[:x\- ]\s*(\d+)',
    r'\b(\d+)\s*x\b',
]

def extract_ipq(text: str) -> int:
    if not isinstance(text, str): return 1
    t = text.lower()
    for pat in IPQ_PATTERNS:
        m = re.search(pat, t)
        if m:
            try:
                v = int(m.group(1))
                return max(1, v)
            except:
                pass
    return 1

def extract_brand(text: str) -> str:
    if not isinstance(text, str): return 'unknown'
    # take first capitalized span as brand guess
    m = _BRAND_RE.search(text.strip())
    if not m:
        return 'unknown'
    brand = m.group(1).strip()
    brand = re.sub(r'[\u00AE\u2122]', '', brand)  # remove ® ™
    brand = re.sub(r'[^A-Za-z0-9&\-\' ]','', brand).strip()
    return brand.lower() if brand else 'unknown'

def parse_units(text: str):
    if not isinstance(text, str): return {}
    t = text.lower()
    out = {}
    for name, factor, pat in _UNITS:
        m = re.search(pat, t)
        if m:
            try:
                qty = float(m.group(1)) * factor
                out[name] = qty
            except:
                pass
    return out

def basic_flags(text: str):
    t = str(text).lower()
    flags = {
        'is_bundle': int(any(w in t for w in ['bundle','combo','set of','pack of','x'])),
        'is_refill': int('refill' in t or 'refills' in t),
        'is_mini': int(any(w in t for w in ['mini','travel size','trial'])),
        'is_premium': int(any(w in t for w in ['premium','luxury','pro','ultra'])),
    }
    return flags

# ---------- vectorizer ----------
def build_tfidf(train_texts, test_texts, max_features=50_000):
    # Reduced features to avoid memory issues
    # both word & char ngrams
    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1,2),
        min_df=10,  # Increased from 5 to reduce features
        max_features=max_features//2,
        sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(3,4),  # Reduced from (3,5) to save memory
        min_df=10,  # Increased from 5 to reduce features
        max_features=max_features//2,
        sublinear_tf=True
    )
    Xw_tr = word_vec.fit_transform(train_texts)
    Xw_te = word_vec.transform(test_texts)
    Xc_tr = char_vec.fit_transform(train_texts)
    Xc_te = char_vec.transform(test_texts)
    from scipy.sparse import hstack
    return hstack([Xw_tr, Xc_tr]), hstack([Xw_te, Xc_te]), word_vec, char_vec

def make_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    ipq = df['catalog_content'].apply(extract_ipq).astype(float)
    unit_info = df['catalog_content'].apply(parse_units)
    qty_ml = unit_info.apply(lambda d: d.get('ml', d.get('l', np.nan)))
    qty_g  = unit_info.apply(lambda d: d.get('g',  d.get('kg', np.nan)))
    qty_cm = unit_info.apply(lambda d: d.get('cm', d.get('inch', np.nan)))
    brand = df['catalog_content'].apply(extract_brand)
    flags = df['catalog_content'].apply(basic_flags)
    flags_df = pd.DataFrame(list(flags))
    num = pd.DataFrame({
        'len_chars': df['catalog_content'].astype(str).apply(len),
        'len_words': df['catalog_content'].astype(str).apply(lambda x: len(x.split())),
        'digit_ratio': df['catalog_content'].astype(str).apply(lambda x: sum(ch.isdigit() for ch in x)/max(1,len(x))),
        'ipq': ipq,
        'log_ipq': np.log1p(ipq),
        'qty_ml': qty_ml.fillna(0.0),
        'qty_g':  qty_g.fillna(0.0),
        'qty_cm': qty_cm.fillna(0.0),
        'brand_norm': brand,
    }).reset_index(drop=True)
    num = pd.concat([num, flags_df.reset_index(drop=True)], axis=1)
    return num
