import pandas as pd
import numpy as np

def load_csv(path):
    return pd.read_csv(path)

def criteria_columns(df):
    meta = {'id','name','label','class','pref','r_id','benefit','cost'}
    return [c for c in df.columns if c not in meta]

def apply_benefit_cost(X, benefit_mask):
    """
    Zamienia kryteria typu COST na –X (bo wszystkie metody zakładają benefit).
    benefit_mask: lista True/False
    """
    X2 = X.copy()
    for i, is_benefit in enumerate(benefit_mask):
        if not is_benefit:
            X2[:, i] = -X2[:, i]
    return X2

def normalize_minmax(X, mins=None, maxs=None):
    X = np.asarray(X, dtype=float)
    if mins is None:
        mins = X.min(axis=0)
    if maxs is None:
        maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    Xn = (X - mins) / ranges
    return Xn, mins, maxs
