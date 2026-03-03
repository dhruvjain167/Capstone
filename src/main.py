import json
import pandas as pd

from src.load_asset import load_all_assets, compute_returns
from src.sentiment_engine import FinBERTSentiment
from src.backtester import run_backtest
from src.evaluation import (
    sharpe_ratio,
    sortino_ratio,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    calmar_ratio,
    value_at_risk,
    hedge_effectiveness,
    variance_reduction_significance,
    hedge_effectiveness_uplift_pct,
)
from config import TARGET_ASSET, BOOTSTRAP_SAMPLES, RANDOM_SEED


def _load_sentiment(returns, days=14):
    print("Fetching market sentiment...")
    try:
        sent_engine = FinBERTSentiment()
        daily_sentiment = sent_engine.get_daily_sentiment(days=days)
    except Exception as exc:
        print(f"Sentiment pipeline unavailable ({exc}). Using zero sentiment.")
        daily_sentiment = pd.DataFrame()

    if daily_sentiment.empty:
        returns["sentiment_score"] = 0.0
        return returns

    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    daily_sentiment.set_index("date", inplace=True)

    returns = returns.join(daily_sentiment, how="left")
    returns["sentiment_score"] = returns["sentiment_score"].fillna(0.0)
    return returns


def _compute_metrics(unhedged, hedged, diagnostics):
    p_val, delta_mean, delta_ci = variance_reduction_significance(
        unhedged,
        hedged,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=RANDOM_SEED,
    )

    metrics = {
        "sharpe_ratio": float(sharpe_ratio(hedged)),
        "sortino_ratio": float(sortino_ratio(hedged)),
        "annualized_return": float(annualized_return(hedged)),
        "annualized_volatility": float(annualized_volatility(hedged)),
        "max_drawdown": float(max_drawdown(hedged)),
        "calmar_ratio": float(calmar_ratio(hedged)),
        "var_95": float(value_at_risk(hedged, q=0.05)),
        "hedge_effectiveness": float(hedge_effectiveness(unhedged, hedged)),
        "variance_reduction_p_value": float(p_val),
        "variance_reduction_mean_delta": float(delta_mean),
        "variance_reduction_ci_low": float(delta_ci[0]),
        "variance_reduction_ci_high": float(delta_ci[1]),
    }

    if not diagnostics.empty:
        if "raw_hedged_return" in diagnostics.columns:
            metrics["annualized_return_gross_pre_blend"] = float(annualized_return(diagnostics["raw_hedged_return"].values))
        if "safe_allocation" in diagnostics.columns:
            metrics["avg_safe_allocation"] = float(diagnostics["safe_allocation"].mean())
        if "turnover" in diagnostics.columns:
            metrics["avg_turnover"] = float(diagnostics["turnover"].mean())
        if "hedge_quality_scale" in diagnostics.columns:
            metrics["avg_hedge_quality_scale"] = float(diagnostics["hedge_quality_scale"].mean())
        if "regime_label" in diagnostics.columns:
            metrics["pct_stress_regime"] = float((diagnostics["regime_label"] == "stress").mean())
        if "regime_obs" in diagnostics.columns:
            metrics["avg_regime_obs"] = float(diagnostics["regime_obs"].mean())

    return metrics


def main():
    print("Loading asset prices...")
    prices = load_all_assets()

    print("Computing returns...")
    returns = compute_returns(prices)
    returns.index = pd.to_datetime(returns.index)

    returns = _load_sentiment(returns, days=14)

    print("\nRunning baseline backtest...")
    baseline_returns, baseline_diag = run_backtest(
        returns,
        return_diagnostics=True,
        use_defensive_overlay=False,
        use_quality_controls=False,
    )

    print("Running enhanced backtest...")
    enhanced_returns, enhanced_diag = run_backtest(
        returns,
        return_diagnostics=True,
        use_defensive_overlay=True,
        use_quality_controls=True,
    )

    unhedged_base = returns[TARGET_ASSET].iloc[-len(baseline_returns) :]
    unhedged_enh = returns[TARGET_ASSET].iloc[-len(enhanced_returns) :]

    baseline_metrics = _compute_metrics(unhedged_base, baseline_returns, baseline_diag)
    enhanced_metrics = _compute_metrics(unhedged_enh, enhanced_returns, enhanced_diag)

    uplift = hedge_effectiveness_uplift_pct(
        baseline_metrics["hedge_effectiveness"],
        enhanced_metrics["hedge_effectiveness"],
    )
    enhanced_metrics["baseline_hedge_effectiveness"] = float(baseline_metrics["hedge_effectiveness"])
    enhanced_metrics["hedge_effectiveness_uplift_pct"] = float(uplift)
    enhanced_metrics["target_uplift_10pct_achieved"] = bool(uplift >= 0.10)

    use_enhanced = (
        enhanced_metrics["hedge_effectiveness"] >= baseline_metrics["hedge_effectiveness"]
        and (
            enhanced_metrics["variance_reduction_p_value"] <= baseline_metrics["variance_reduction_p_value"]
            or enhanced_metrics["hedge_effectiveness"] > 0
        )
    )

    chosen_returns = enhanced_returns if use_enhanced else baseline_returns
    chosen_diag = enhanced_diag if use_enhanced else baseline_diag
    chosen_metrics = enhanced_metrics if use_enhanced else baseline_metrics
    chosen_metrics["selected_strategy"] = "enhanced" if use_enhanced else "baseline"

    print("\n========== PERFORMANCE ==========")
    print("Selected strategy:", chosen_metrics["selected_strategy"])
    print("Sharpe Ratio:", round(chosen_metrics["sharpe_ratio"], 4))
    print("Sortino Ratio:", round(chosen_metrics["sortino_ratio"], 4))
    print("Max Drawdown:", round(chosen_metrics["max_drawdown"], 4))
    print("Hedge Effectiveness:", round(chosen_metrics["hedge_effectiveness"], 4))
    print("Variance-reduction p-value:", round(chosen_metrics["variance_reduction_p_value"], 4))
    if "hedge_effectiveness_uplift_pct" in chosen_metrics:
        print("Hedge effectiveness uplift vs baseline:", round(chosen_metrics["hedge_effectiveness_uplift_pct"] * 100, 2), "%")

    results_df = pd.DataFrame({"Hedged_Return": chosen_returns})
    results_df.to_csv("hedged_portfolio_results.csv", index=False)

    if not chosen_diag.empty:
        chosen_diag.to_csv("hedge_diagnostics.csv", index=False)

    with open("performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(chosen_metrics, f, indent=2)

    print("\nResults saved to hedged_portfolio_results.csv")
    print("Diagnostics saved to hedge_diagnostics.csv and performance_metrics.json")


if __name__ == "__main__":
    main()
