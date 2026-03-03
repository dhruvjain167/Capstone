import numpy as np


def sharpe_ratio(r):
    denom = np.std(r)
    if denom <= 1e-12:
        return 0.0
    return np.mean(r) / denom


def sortino_ratio(r):
    downside = np.std(np.minimum(r, 0.0))
    if downside <= 1e-12:
        return 0.0
    return np.mean(r) / downside


def annualized_return(r, periods_per_year=252):
    if len(r) == 0:
        return 0.0
    growth = np.prod(1 + r)
    years = len(r) / periods_per_year
    if years <= 0:
        return 0.0
    return growth ** (1 / years) - 1


def annualized_volatility(r, periods_per_year=252):
    return np.std(r) * np.sqrt(periods_per_year)


def max_drawdown(r):
    cumulative = (1 + r).cumprod()
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calmar_ratio(r, periods_per_year=252):
    mdd = abs(max_drawdown(r))
    if mdd <= 1e-12:
        return 0.0
    return annualized_return(r, periods_per_year=periods_per_year) / mdd


def value_at_risk(r, q=0.05):
    if len(r) == 0:
        return 0.0
    return float(np.quantile(r, q))


def hedge_effectiveness(unhedged, hedged):
    unhedged_var = np.var(unhedged)
    if unhedged_var <= 1e-12:
        return 0.0
    return 1 - (np.var(hedged) / unhedged_var)


def hedge_effectiveness_uplift_pct(baseline_eff, improved_eff):
    base = max(abs(baseline_eff), 1e-8)
    return float((improved_eff - baseline_eff) / base)


def variance_reduction_significance(unhedged, hedged, n_bootstrap=2000, seed=42):
    """
    Bootstrap test on daily variance reduction delta:
    delta_t = unhedged_t^2 - hedged_t^2.
    p-value is one-sided probability that mean(delta)<=0.
    """
    unhedged = np.asarray(unhedged, dtype=float)
    hedged = np.asarray(hedged, dtype=float)
    n = min(len(unhedged), len(hedged))
    if n == 0:
        return 1.0, 0.0, (0.0, 0.0)

    u = unhedged[-n:]
    h = hedged[-n:]
    delta = (u**2) - (h**2)
    mean_delta = float(np.mean(delta))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples = delta[idx].mean(axis=1)

    p_value = float(np.mean(samples <= 0.0))
    ci = (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
    return p_value, mean_delta, ci
