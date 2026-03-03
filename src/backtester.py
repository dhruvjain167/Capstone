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
    SAFE_ASSET,
    MIN_SAFE_ALLOCATION,
    MAX_SAFE_ALLOCATION,
    MOMENTUM_LOOKBACK,
    DRAWDOWN_LOOKBACK,
    DRAWDOWN_THRESHOLD,
    HEDGE_SHRINKAGE,
    CORRELATION_FLOOR,
    NO_TRADE_BAND,
    STABILITY_LOOKBACK,
    REGIME_METHOD,
    REGIME_LOOKBACK,
    REGIME_VOL_MULTIPLIER,
    REGIME_MIN_OBS,
    DCC_A_CALM,
    DCC_B_CALM,
    DCC_A_STRESS,
    DCC_B_STRESS,
    STRESS_GROSS_EXPOSURE_MULT,
    STRESS_HEDGE_SHRINKAGE_MULT,
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


def _rolling_momentum(series, lookback):
    if len(series) < lookback:
        return 0.0
    return float(np.prod(1 + series[-lookback:]) - 1)


def _portfolio_drawdown(recent_returns):
    if len(recent_returns) == 0:
        return 0.0
    cumulative = np.cumprod(1 + np.asarray(recent_returns, dtype=float))
    peaks = np.maximum.accumulate(cumulative)
    dd = (cumulative - peaks) / np.maximum(peaks, 1e-12)
    return float(dd.min())


def _safe_allocation(sentiment_t, target_window, recent_portfolio_returns):
    sentiment_penalty = max(float(np.tanh(-sentiment_t)), 0.0)
    momentum = _rolling_momentum(target_window.values, MOMENTUM_LOOKBACK)
    momentum_penalty = 1.0 if momentum < 0 else 0.0

    if len(target_window) >= MOMENTUM_LOOKBACK:
        vol_now = np.std(target_window.values[-MOMENTUM_LOOKBACK:])
    else:
        vol_now = np.std(target_window.values)

    vol_baseline = np.std(target_window.values)
    vol_stress = np.clip((vol_now / max(vol_baseline, 1e-8)) - 1.0, 0.0, 1.0)

    lookback_returns = recent_portfolio_returns[-DRAWDOWN_LOOKBACK:]
    drawdown = _portfolio_drawdown(lookback_returns)
    drawdown_penalty = np.clip(abs(min(drawdown, 0.0)) / abs(DRAWDOWN_THRESHOLD), 0.0, 1.0)

    stress_score = 0.35 * sentiment_penalty + 0.30 * momentum_penalty + 0.20 * vol_stress + 0.15 * drawdown_penalty
    safe_w = MIN_SAFE_ALLOCATION + (MAX_SAFE_ALLOCATION - MIN_SAFE_ALLOCATION) * np.clip(stress_score, 0.0, 1.0)

    return float(np.clip(safe_w, MIN_SAFE_ALLOCATION, MAX_SAFE_ALLOCATION)), float(stress_score), float(drawdown)


def _hedge_quality_scale(target_window, hedge_window):
    """Use realized target-vs-hedge co-movement to avoid unstable hedge ratios."""
    if len(target_window) < 5 or len(hedge_window) < 5:
        return 0.0, 0.0

    corr = np.corrcoef(target_window, hedge_window)[0, 1]
    if not np.isfinite(corr):
        return 0.0, 0.0

    if corr <= CORRELATION_FLOOR:
        return 0.0, float(corr)

    quality = np.clip((corr - CORRELATION_FLOOR) / max(1.0 - CORRELATION_FLOOR, 1e-8), 0.0, 1.0)
    return float(quality), float(corr)


def _apply_no_trade_band(candidate_weights, prev_weights):
    delta = candidate_weights - prev_weights
    keep_prev = np.abs(delta) < NO_TRADE_BAND
    final = np.where(keep_prev, prev_weights, candidate_weights)
    return final, int(np.sum(keep_prev))


def _detect_regime_flags(target_series):
    """Volatility-threshold regime detector: True means stress regime."""
    x = np.asarray(target_series, dtype=float)
    n = len(x)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags

    look = max(5, min(REGIME_LOOKBACK, n))
    for i in range(look, n + 1):
        recent = x[i - look : i]
        long_window = x[:i]
        vol_recent = np.std(recent)
        vol_long = np.std(long_window)
        if vol_recent > REGIME_VOL_MULTIPLIER * max(vol_long, 1e-8):
            flags[i - 1] = True
    return flags


def _regime_dcc(std_resids_array, regime_flags):
    current_is_stress = bool(regime_flags[-1]) if len(regime_flags) else False

    if current_is_stress:
        dcc_a, dcc_b = DCC_A_STRESS, DCC_B_STRESS
        label = "stress"
    else:
        dcc_a, dcc_b = DCC_A_CALM, DCC_B_CALM
        label = "calm"

    # Estimate separate DCC for the current regime using matching history.
    idx = np.where(regime_flags == current_is_stress)[0]
    if len(idx) >= REGIME_MIN_OBS:
        resids_for_dcc = std_resids_array[idx]
        used_obs = int(len(idx))
    else:
        # fallback to full sample if regime-specific history is too short
        resids_for_dcc = std_resids_array
        used_obs = int(len(std_resids_array))

    R_series = compute_dcc(resids_for_dcc, dcc_a, dcc_b)
    return R_series[-1], label, dcc_a, dcc_b, used_obs


def run_backtest(df, return_diagnostics=False, use_defensive_overlay=True, use_quality_controls=True):
    if "sentiment_score" not in df.columns:
        df = df.copy()
        df["sentiment_score"] = 0.0

    available_hedges = [a for a in HEDGE_ASSETS if a in df.columns]
    ordered_assets = [TARGET_ASSET] + available_hedges

    returns = df[ordered_assets]
    sentiment = df["sentiment_score"].fillna(0.0)
    safe_returns = df[SAFE_ASSET].fillna(0.0) if SAFE_ASSET in df.columns else pd.Series(0.0, index=df.index)

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

        regime_flags = _detect_regime_flags(window[TARGET_ASSET].fillna(0.0).values)

        if REGIME_METHOD == "vol_threshold":
            R_t, regime_label, regime_dcc_a, regime_dcc_b, regime_obs = _regime_dcc(std_resids_array, regime_flags)
        else:
            # default fallback to non-regime DCC
            R_series = compute_dcc(std_resids_array, DCC_A, DCC_B)
            R_t = R_series[-1]
            regime_label, regime_dcc_a, regime_dcc_b, regime_obs = "single", DCC_A, DCC_B, len(std_resids_array)

        H_t = engine.compute_covariance_matrix(R_t, sigmas_last)

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

        smoothed_weights = WEIGHT_SMOOTHING_ALPHA * full_new + (1.0 - WEIGHT_SMOOTHING_ALPHA) * prev_weights

        regime_shrink = HEDGE_SHRINKAGE
        regime_gross_cap = MAX_GROSS_EXPOSURE
        if regime_label == "stress":
            regime_shrink = HEDGE_SHRINKAGE * STRESS_HEDGE_SHRINKAGE_MULT
            regime_gross_cap = MAX_GROSS_EXPOSURE * STRESS_GROSS_EXPOSURE_MULT

        if use_quality_controls:
            smoothed_weights = regime_shrink * smoothed_weights
            lookback_start = max(0, t - STABILITY_LOOKBACK)
            target_window = returns[TARGET_ASSET].iloc[lookback_start:t].values
            hedge_window = np.dot(
                returns[available_hedges].iloc[lookback_start:t].fillna(0.0).values,
                np.nan_to_num(smoothed_weights, nan=0.0),
            )
            quality_scale, realized_corr = _hedge_quality_scale(target_window, hedge_window)
            quality_weights = smoothed_weights * quality_scale
            capped_weights, gross_exposure = _normalize_to_gross_cap(quality_weights, regime_gross_cap)
            final_weights, no_trade_assets = _apply_no_trade_band(capped_weights, prev_weights)
        else:
            quality_scale, realized_corr, no_trade_assets = 1.0, 0.0, 0
            final_weights, gross_exposure = _normalize_to_gross_cap(smoothed_weights, regime_gross_cap)

        turnover = float(np.sum(np.abs(final_weights - prev_weights)))
        cost = turnover * (TRANSACTION_COST_BPS / 10000.0)

        r_t = returns.iloc[t]
        hedge_asset_returns = np.nan_to_num(r_t[available_hedges].values, nan=0.0, posinf=0.0, neginf=0.0)
        target_return = float(r_t.get(TARGET_ASSET, 0.0))

        hedge_returns = float(np.dot(final_weights, hedge_asset_returns))
        raw_portfolio_r = target_return - hedge_returns - cost

        if use_defensive_overlay:
            safe_weight, stress_score, rolling_drawdown = _safe_allocation(
                sentiment.iloc[t],
                returns[TARGET_ASSET].iloc[t - WINDOW : t],
                portfolio_returns,
            )
            if regime_label == "stress":
                safe_weight = float(np.clip(safe_weight + 0.10, MIN_SAFE_ALLOCATION, MAX_SAFE_ALLOCATION))
            safe_return = float(safe_returns.iloc[t])
        else:
            safe_weight, stress_score, rolling_drawdown = 0.0, 0.0, 0.0
            safe_return = 0.0

        blended_portfolio_r = (1.0 - safe_weight) * raw_portfolio_r + safe_weight * safe_return

        est_vol = _ewma_volatility(portfolio_returns, lam=EWMA_LAMBDA)
        leverage = float(np.clip(VOL_TARGET / max(est_vol, 1e-6), 0.75, 1.25))
        portfolio_r = blended_portfolio_r * leverage

        portfolio_returns.append(portfolio_r)

        abs_sum = float(np.sum(np.abs(final_weights)))
        alloc = np.abs(final_weights) / max(abs_sum, 1e-12)

        diagnostics_row = {
            "date": returns.index[t],
            "target_return": target_return,
            "hedge_return": hedge_returns,
            "transaction_cost": cost,
            "raw_hedged_return": raw_portfolio_r,
            "blended_return_pre_leverage": blended_portfolio_r,
            "leverage": leverage,
            "hedged_return": portfolio_r,
            "safe_asset": SAFE_ASSET,
            "safe_asset_return": safe_return,
            "safe_allocation": safe_weight,
            "regime_stress": stress_score,
            "rolling_drawdown": rolling_drawdown,
            "realized_target_hedge_corr": realized_corr,
            "hedge_quality_scale": quality_scale,
            "no_trade_assets": no_trade_assets,
            "regime_label": regime_label,
            "regime_dcc_a": regime_dcc_a,
            "regime_dcc_b": regime_dcc_b,
            "regime_obs": regime_obs,
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
