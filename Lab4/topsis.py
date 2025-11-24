import numpy as np

def topsis_score(X_raw, weights=None, benefit_mask=None, norm='l2'):
    """
    X_raw – dane alternatyw
    benefit_mask – lista True/False
    """
    X = X_raw.copy()

    # cost → multiply by -1
    if benefit_mask is not None:
        for i, b in enumerate(benefit_mask):
            if not b:
                X[:, i] *= -1

    m, n = X.shape
    if weights is None:
        w = np.ones(n)
    else:
        w = np.array(weights, dtype=float)

    # normalize
    denom = np.sqrt((X**2).sum(axis=0))
    denom[denom == 0] = 1
    Xn = X / denom

    # weight
    Xw = Xn * w

    # ideal & anti-ideal
    v_pos = Xw.max(axis=0)
    v_neg = Xw.min(axis=0)

    d_pos = np.sqrt(((Xw - v_pos)**2).sum(axis=1))
    d_neg = np.sqrt(((Xw - v_neg)**2).sum(axis=1))

    score = d_neg / (d_pos + d_neg + 1e-9)
    return score, {"Xw": Xw, "v_pos": v_pos, "v_neg": v_neg}
