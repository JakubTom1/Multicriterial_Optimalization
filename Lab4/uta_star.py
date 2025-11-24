import numpy as np
import pulp

def build_grid_vals(x, P):
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.ones(P)*mn
    return np.linspace(mn, mx, P)

def compute_alphas(xvals, grid):
    m = len(xvals)
    P = len(grid)
    A = np.zeros((m,P))
    for i,x in enumerate(xvals):
        if x <= grid[0]:
            A[i,0] = 1
        elif x >= grid[-1]:
            A[i,-1] = 1
        else:
            idx = np.searchsorted(grid, x)
            if grid[idx] == x:
                A[i,idx] = 1
            else:
                p1 = idx-1; p2=idx
                g1,g2 = grid[p1], grid[p2]
                t = (x-g1)/(g2-g1)
                A[i,p1]=1-t; A[i,p2]=t
    return A

def fit_uta_star(df, criteria, P=6, rank_col="rank", eps=1e-3):
    X = df[criteria].to_numpy(float)
    m,n = X.shape

    grids = [build_grid_vals(X[:,j], P) for j in range(n)]
    alphas = [compute_alphas(X[:,j], grids[j]) for j in range(n)]

    prob = pulp.LpProblem("UTA", pulp.LpMinimize)

    g = {(j,p): pulp.LpVariable(f"g_{j}_{p}", lowBound=0, upBound=1)
         for j in range(n) for p in range(P)}

    z = {j: pulp.LpVariable(f"z_{j}", lowBound=0, upBound=1)
         for j in range(n)}

    for j in range(n):
        prob += g[(j,P-1)] == z[j]

    for j in range(n):
        for p in range(1,P):
            prob += g[(j,p)] >= g[(j,p-1)]

    prob += pulp.lpSum(z.values()) == 1

    u = [pulp.LpVariable(f"u_{i}", lowBound=0) for i in range(m)]

    for i in range(m):
        terms=[]
        for j in range(n):
            A = alphas[j][i]
            for p in range(P):
                if A[p]>0:
                    terms.append(A[p]*g[(j,p)])
        prob += u[i] == pulp.lpSum(terms) if terms else u[i] == 0

    sl=[]
    ranks=df[rank_col].to_numpy(int)
    for i in range(m):
        for k in range(m):
            if ranks[i]<ranks[k]:
                s=pulp.LpVariable(f"s_{i}_{k}",lowBound=0)
                sl.append(s)
                prob += u[i] >= u[k] + eps - s

    prob += pulp.lpSum(sl)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    gsol={j: np.array([pulp.value(g[(j,p)]) for p in range(P)]) for j in range(n)}
    usol=np.array([pulp.value(ui) for ui in u])
    usol=(usol-usol.min())/(usol.max()-usol.min()+1e-9)

    return {"grids":grids,"g":gsol,"criteria":criteria}, usol
