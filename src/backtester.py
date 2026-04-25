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
    SAFE_ASSET,
    MIN_TARGET_EXPOSURE,
    MAX_TARGET_EXPOSURE,
    SENTIMENT_EXPOSURE_BETA,
    MOMENTUM_EXPOSURE_BETA,
    DRAWDOWN_EXPOSURE_BETA,
    CASH_DAILY_RETURN_FLOOR,
    RECENT_DRAWDOWN_WINDOW,
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


def _recent_drawdown(returns_window):
    if len(returns_window) < 2:
        return 0.0

    cumulative = np.cumprod(1 + np.array(returns_window, dtype=float))
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_peak) / np.maximum(running_peak, 1e-12)
    return float(np.min(drawdowns))


def _classify_regime(window_target, est_vol, vol_threshold_high=0.015, vol_threshold_low=0.006):
    """
    Classify market regime based on volatility and momentum.
    Returns: 'crisis', 'high_vol', 'normal', 'low_vol'
    """
    momentum = float(window_target.mean())
    
    if est_vol > vol_threshold_high and momentum < 0:
        return "crisis"
    elif est_vol > vol_threshold_high:
        return "high_vol"
    elif est_vol < vol_threshold_low:
        return "low_vol"
    else:
        return "normal"


def _target_exposure(window_target, sentiment_value, recent_portfolio_returns):
    momentum = float(window_target.mean())
    momentum_vol = float(window_target.std())
    momentum_score = momentum / max(momentum_vol, 1e-6)

    sentiment_score = float(np.clip(sentiment_value, -1.0, 1.0))
    drawdown = _recent_drawdown(recent_portfolio_returns[-RECENT_DRAWDOWN_WINDOW:])

    raw_exposure = (
        0.65
        + MOMENTUM_EXPOSURE_BETA * np.tanh(momentum_score)
        + SENTIMENT_EXPOSURE_BETA * np.tanh(sentiment_score)
        + DRAWDOWN_EXPOSURE_BETA * np.tanh(drawdown * 5)
    )

    return float(np.clip(raw_exposure, MIN_TARGET_EXPOSURE, MAX_TARGET_EXPOSURE))


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
        hedge_vector, sentiment_multiplier = engine.adjust_for_sentiment(hedge_vector, sentiment.iloc[t])

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
        exposure = _target_exposure(window[TARGET_ASSET], sentiment.iloc[t], portfolio_returns)
        safe_weight = 1.0 - exposure

        safe_return = float(r_t.get(SAFE_ASSET, np.nan)) if SAFE_ASSET in returns.columns else np.nan
        if not np.isfinite(safe_return):
            safe_return = CASH_DAILY_RETURN_FLOOR
        safe_return = max(safe_return, CASH_DAILY_RETURN_FLOOR)

        blended_return = (exposure * raw_portfolio_r) + (safe_weight * safe_return)
        portfolio_r = blended_return * leverage

        portfolio_returns.append(portfolio_r)

        # Classify market regime
        regime = _classify_regime(window[TARGET_ASSET], est_vol)

        # Compute confidence scale for diagnostics
        confidence = engine.confidence_scale(H_t)

        # Build rolling correlation between target and each hedge asset
        corr_dict = {}
        for i, asset_name in enumerate(hedge_assets):
            if asset_name in window.columns:
                corr_val = window[TARGET_ASSET].corr(window[asset_name])
                corr_dict[f"corr_{asset_name}"] = float(corr_val) if np.isfinite(corr_val) else 0.0

        diag_entry = {
            "date": returns.index[t],
            "target_return": float(r_t.get(TARGET_ASSET, 0.0)),
            "hedge_return": hedge_returns,
            "raw_hedged_return": raw_portfolio_r,
            "blended_return": blended_return,
            "leverage": leverage,
            "hedged_return": portfolio_r,
            "sentiment_score": float(sentiment.iloc[t]),
            "sentiment_multiplier": sentiment_multiplier,
            "target_exposure": exposure,
            "safe_asset_weight": safe_weight,
            "safe_asset_return": safe_return,
            "est_vol": est_vol,
            "regime": regime,
            "confidence": confidence,
        }
        
        # Add correlations
        diag_entry.update(corr_dict)

        diagnostics.append(diag_entry)

        if len(hedge_assets) > 0:
            for asset_name, weight in zip(hedge_assets, trimmed_hedge):
                diagnostics[-1][f"weight_{asset_name}"] = float(weight)

    result = np.array(portfolio_returns) if portfolio_returns else np.array([0.0])

    if return_diagnostics:
        diagnostics_df = pd.DataFrame(diagnostics)
        return result, diagnostics_df

    return result
