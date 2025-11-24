import numpy as np
from scipy.spatial.distance import cdist

def mref_score(X, refs, prefs, method='inverse_distance', sigma=None, eps=1e-6):
    X = np.asarray(X, dtype=float)
    refs = np.asarray(refs, dtype=float)
    prefs = np.asarray(prefs, dtype=float)

    D = cdist(X, refs, metric='euclidean')

    if method == 'nearest':
        idx = np.argmin(D, axis=1)
        return prefs[idx], {"D": D, "nearest": idx}

    if method == 'inverse_distance':
        W = 1.0 / (D + eps)
    elif method == 'gaussian':
        if sigma is None:
            sigma = np.median(D) + eps
        W = np.exp(-(D/sigma)**2)
    else:
        raise ValueError("Unknown method.")

    Wsum = W.sum(axis=1, keepdims=True)
    Wsum[Wsum == 0] = 1
    Wn = W / Wsum

    S = Wn.dot(prefs)
    # normalize
    S = (S - S.min())/(S.max()-S.min()+1e-9)
    return S, {"D": D, "W": W, "Wn": Wn}
