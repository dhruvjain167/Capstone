import numpy as np
import pandas as pd

from config import (
    WINDOW,
    STEP,
    DCC_A,
    DCC_B,
    TARGET_ASSET,
    RIDGE_LAMBDA,
    MAX_HEDGE_WEIGHT,
    VOL_TARGET,
    EWMA_LAMBDA,
)
from .garch_x_model import fit_garch, fit_garch_x
from .dcc_model import compute_dcc
from .hedge_engine import HedgeEngine


def _ewma_volatility(values, lam=0.94):
    if len(values) < 2:
        return VOL_TARGET

    vol2 = np.var(values[: min(20, len(values))])
    for x in values:
        vol2 = lam * vol2 + (1 - lam) * (x ** 2)

    return float(np.sqrt(max(vol2, 1e-10)))


def run_backtest(df, return_diagnostics=False):
    if "sentiment_score" not in df.columns:
        df = df.copy()
        df["sentiment_score"] = 0.0

    ordered_assets = [TARGET_ASSET] + [c for c in df.columns if c not in {TARGET_ASSET, "sentiment_score"}]
    returns = df[ordered_assets]
    sentiment = df["sentiment_score"]

    engine = HedgeEngine()
    portfolio_returns = []
    diagnostics = []

    for t in range(WINDOW, len(df), STEP):
        window = returns.iloc[t - WINDOW : t]
        window_sent = sentiment.iloc[t - WINDOW : t].shift(1).fillna(0.0)

        sigmas_last = []
        std_resids = []
        valid_cols = []

        for col in window.columns:
            if window[col].isna().all():
                continue

            valid_cols.append(col)

            if col == TARGET_ASSET:
                sigma, resid = fit_garch_x(window[col], window_sent)
            else:
                sigma, resid = fit_garch(window[col])

            last_sigma = sigma.iloc[-1]
            if pd.isna(last_sigma) or np.isinf(last_sigma):
                last_sigma = 0.01
            sigmas_last.append(last_sigma)

            resid_array = np.nan_to_num(resid.values.flatten(), nan=0.0, posinf=0.01, neginf=-0.01)
            std_resids.append(resid_array)

        if len(std_resids) < 2 or TARGET_ASSET not in valid_cols:
            continue

        std_resids_array = np.column_stack(std_resids)
        if np.all(std_resids_array == 0) or np.all(np.isnan(std_resids_array)):
            continue

        R_series = compute_dcc(std_resids_array, DCC_A, DCC_B)
        H_t = engine.compute_covariance_matrix(R_series[-1], sigmas_last)

        hedge_vector = engine.compute_multivariate_hedge(
            H_t,
            ridge_lambda=RIDGE_LAMBDA,
            max_weight=MAX_HEDGE_WEIGHT,
        )
        hedge_vector = engine.adjust_for_sentiment(hedge_vector, sentiment.iloc[t])

        r_t = returns.iloc[t]
        hedge_assets = [c for c in valid_cols if c != TARGET_ASSET]

        if len(hedge_assets) == 0:
            raw_portfolio_r = float(r_t.get(TARGET_ASSET, 0.0))
            hedge_returns = 0.0
        else:
            valid_returns = np.nan_to_num(r_t[hedge_assets].values, nan=0.0, posinf=0.0, neginf=0.0)
            trimmed_hedge = hedge_vector[: len(hedge_assets)]
            hedge_returns = float(np.dot(trimmed_hedge, valid_returns))
            raw_portfolio_r = float(r_t.get(TARGET_ASSET, 0.0) - hedge_returns)

        est_vol = _ewma_volatility(portfolio_returns, lam=EWMA_LAMBDA)
        leverage = float(np.clip(VOL_TARGET / max(est_vol, 1e-6), 0.5, 1.5))
        portfolio_r = raw_portfolio_r * leverage

        portfolio_returns.append(portfolio_r)

        diagnostics.append(
            {
                "date": returns.index[t],
                "target_return": float(r_t.get(TARGET_ASSET, 0.0)),
                "hedge_return": hedge_returns,
                "raw_hedged_return": raw_portfolio_r,
                "leverage": leverage,
                "hedged_return": portfolio_r,
                "sentiment_score": float(sentiment.iloc[t]),
                "est_vol": est_vol,
            }
        )

        if len(hedge_assets) > 0:
            for asset_name, weight in zip(hedge_assets, trimmed_hedge):
                diagnostics[-1][f"weight_{asset_name}"] = float(weight)

    result = np.array(portfolio_returns) if portfolio_returns else np.array([0.0])

    if return_diagnostics:
        diagnostics_df = pd.DataFrame(diagnostics)
        return result, diagnostics_df

    return result
