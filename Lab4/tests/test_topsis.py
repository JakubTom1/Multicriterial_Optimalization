import numpy as np
from Lab4.topsis import topsis_score

def test_topsis_order():
    X = np.array([[1.0, 10.0],
                  [2.0, 5.0]])
    mask = [False, True]
    scores, _ = topsis_score(X, weights=[0.5,0.5], benefit_mask=mask)
    assert scores.shape[0] == 2
