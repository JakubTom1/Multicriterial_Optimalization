import numpy as np

def rdp(points, eps):
    points = np.asarray(points)
    if points.shape[0] < 3:
        return points.copy()

    start, end = points[0], points[-1]
    vec = end - start
    L2 = (vec**2).sum()
    if L2 == 0:
        d = np.linalg.norm(points - start, axis=1)
    else:
        t = np.dot(points - start, vec)/L2
        t = np.clip(t,0,1)
        proj = start + np.outer(t, vec)
        d = np.linalg.norm(points - proj, axis=1)

    idx = np.argmax(d)
    if d[idx] > eps:
        left = rdp(points[:idx+1], eps)
        right= rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])

def point_to_segment_distance(p, a, b):
    ab = b - a
    ap = p - a
    L2 = (ab**2).sum()
    if L2 == 0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab)/L2
    t = np.clip(t,0,1)
    proj = a + t*ab
    return np.linalg.norm(p - proj)

def sp_cs_score(X, poly, eps_simplify=0.01):
    simp = rdp(poly, eps_simplify)
    d = np.array([
        min( point_to_segment_distance(x, simp[i], simp[i+1]) for i in range(len(simp)-1) )
        for x in X
    ])
    s = 1/(1+d)
    s = (s - s.min())/(s.max()-s.min()+1e-9)
    return s, {"simplified": simp, "dist": d}
