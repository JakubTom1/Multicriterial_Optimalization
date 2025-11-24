import pandas as pd
from Lab4.uta_star import fit_uta_star, predict_uta_star

def test_uta_simple():
    df = pd.DataFrame({'a':[0,1,2],'b':[2,1,0]})
    df['rank'] = [3,2,1]
    fit = fit_uta_star(df, ['a','b'], rank_col='rank')
    u = predict_uta_star(fit, df[['a','b']].to_numpy())
    assert len(u) == 3
