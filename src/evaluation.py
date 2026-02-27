import numpy as np


def sharpe_ratio(r):
    denom = np.std(r)
    if denom <= 1e-12:
        return 0.0
    return np.mean(r) / denom


def max_drawdown(r):
    cumulative = (1 + r).cumprod()
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def hedge_effectiveness(unhedged, hedged):
    unhedged_var = np.var(unhedged)
    if unhedged_var <= 1e-12:
        return 0.0
    return 1 - (np.var(hedged) / unhedged_var)
