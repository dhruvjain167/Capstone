import numpy as np

def sharpe_ratio(r):
    return np.mean(r) / np.std(r)

def max_drawdown(r):
    cumulative = (1 + r).cumprod()
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def hedge_effectiveness(unhedged, hedged):
    return 1 - (np.var(hedged) / np.var(unhedged))