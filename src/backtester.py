import numpy as np
import pandas as pd

from config import (
    WINDOW,
    STEP,
    DCC_A,
    DCC_B,
    TARGET_ASSET,
    HEDGE_ASSETS,
    RIDGE_LAMBDA,
    MAX_HEDGE_WEIGHT,
    VOL_TARGET,
    EWMA_LAMBDA,
    WEIGHT_SMOOTHING_ALPHA,
    MIN_CORRELATION_CONFIDENCE,
    MAX_GROSS_EXPOSURE,
    TRANSACTION_COST_BPS,
    SENTIMENT_STRENGTH,
)
from .garch_x_model import fit_garch, fit_garch_x
from .dcc_model import compute_dcc
from .hedge_engine import HedgeEngine


def _ewma_volatility(values, lam=0.94):
    if len(values) < 2:
        return VOL_TARGET

    vol2 = np.var(values[: min(20, len(values))])
    for x in values:
        vol2 = lam * vol2 + (1 - lam) * (x**2)

    return float(np.sqrt(max(vol2, 1e-10)))


def _normalize_to_gross_cap(weights, gross_cap):
    gross = float(np.sum(np.abs(weights)))
    if gross <= gross_cap:
        return weights, gross
    scale = gross_cap / max(gross, 1e-12)
    return weights * scale, gross_cap


def _weights_to_dict(assets, values):
    return {asset: float(val) for asset, val in zip(assets, values)}


def run_backtest(df, return_diagnostics=False):
    if "sentiment_score" not in df.columns:
        df = df.copy()
        df["sentiment_score"] = 0.0

    available_hedges = [a for a in HEDGE_ASSETS if a in df.columns]
    ordered_assets = [TARGET_ASSET] + available_hedges

    returns = df[ordered_assets]
    sentiment = df["sentiment_score"].fillna(0.0)

    engine = HedgeEngine()
    portfolio_returns = []
    diagnostics = []

    prev_weights = np.zeros(len(available_hedges), dtype=float)

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

        valid_hedges = [c for c in valid_cols if c != TARGET_ASSET]
        if len(valid_hedges) == 0:
            continue

        std_resids_array = np.column_stack(std_resids)
        if np.all(std_resids_array == 0) or np.all(np.isnan(std_resids_array)):
            continue

        R_series = compute_dcc(std_resids_array, DCC_A, DCC_B)
        H_t = engine.compute_covariance_matrix(R_series[-1], sigmas_last)

        raw_hedge = engine.compute_multivariate_hedge(
            H_t,
            ridge_lambda=RIDGE_LAMBDA,
            max_weight=MAX_HEDGE_WEIGHT,
        )

        confidence = engine.confidence_scale(H_t, min_confidence=MIN_CORRELATION_CONFIDENCE)
        conf_scaled = raw_hedge * confidence

        sent_adjusted, sentiment_multiplier = engine.adjust_for_sentiment(
            conf_scaled,
            sentiment.iloc[t],
            strength=SENTIMENT_STRENGTH,
        )

        full_new = np.zeros(len(available_hedges), dtype=float)
        full_raw = np.zeros(len(available_hedges), dtype=float)

        for idx, asset in enumerate(available_hedges):
            if asset in valid_hedges:
                v_idx = valid_hedges.index(asset)
                full_new[idx] = sent_adjusted[v_idx]
                full_raw[idx] = raw_hedge[v_idx]

        smoothed_weights = (
            WEIGHT_SMOOTHING_ALPHA * full_new + (1.0 - WEIGHT_SMOOTHING_ALPHA) * prev_weights
        )
        final_weights, gross_exposure = _normalize_to_gross_cap(smoothed_weights, MAX_GROSS_EXPOSURE)

        turnover = float(np.sum(np.abs(final_weights - prev_weights)))
        cost = turnover * (TRANSACTION_COST_BPS / 10000.0)

        r_t = returns.iloc[t]
        hedge_asset_returns = np.nan_to_num(r_t[available_hedges].values, nan=0.0, posinf=0.0, neginf=0.0)
        target_return = float(r_t.get(TARGET_ASSET, 0.0))

        hedge_returns = float(np.dot(final_weights, hedge_asset_returns))
        raw_portfolio_r = target_return - hedge_returns - cost

        est_vol = _ewma_volatility(portfolio_returns, lam=EWMA_LAMBDA)
        leverage = float(np.clip(VOL_TARGET / max(est_vol, 1e-6), 0.5, 1.5))
        portfolio_r = raw_portfolio_r * leverage

        portfolio_returns.append(portfolio_r)

        abs_sum = float(np.sum(np.abs(final_weights)))
        alloc = np.abs(final_weights) / max(abs_sum, 1e-12)

        diagnostics_row = {
            "date": returns.index[t],
            "target_return": target_return,
            "hedge_return": hedge_returns,
            "transaction_cost": cost,
            "raw_hedged_return": raw_portfolio_r,
            "leverage": leverage,
            "hedged_return": portfolio_r,
            "sentiment_score": float(sentiment.iloc[t]),
            "sentiment_multiplier": sentiment_multiplier,
            "confidence_scale": confidence,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "est_vol": est_vol,
        }

        for asset, val in _weights_to_dict(available_hedges, full_raw).items():
            diagnostics_row[f"weight_raw_{asset}"] = val
        for asset, val in _weights_to_dict(available_hedges, final_weights).items():
            diagnostics_row[f"weight_final_{asset}"] = val
        for asset, val in _weights_to_dict(available_hedges, alloc).items():
            diagnostics_row[f"alloc_{asset}"] = val

        diagnostics.append(diagnostics_row)
        prev_weights = final_weights.copy()

    result = np.array(portfolio_returns) if portfolio_returns else np.array([0.0])

    if return_diagnostics:
        diagnostics_df = pd.DataFrame(diagnostics)
        return result, diagnostics_df

    return result
