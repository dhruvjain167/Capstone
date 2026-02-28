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
)
from config import TARGET_ASSET


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


def _compute_metrics(unhedged, hedged):
    return {
        "sharpe_ratio": float(sharpe_ratio(hedged)),
        "sortino_ratio": float(sortino_ratio(hedged)),
        "annualized_return": float(annualized_return(hedged)),
        "annualized_volatility": float(annualized_volatility(hedged)),
        "max_drawdown": float(max_drawdown(hedged)),
        "calmar_ratio": float(calmar_ratio(hedged)),
        "var_95": float(value_at_risk(hedged, q=0.05)),
        "hedge_effectiveness": float(hedge_effectiveness(unhedged, hedged)),
    }


def main():
    print("Loading asset prices...")
    prices = load_all_assets()

    print("Computing returns...")
    returns = compute_returns(prices)
    returns.index = pd.to_datetime(returns.index)

    returns = _load_sentiment(returns, days=14)

    print("\nRunning dynamic hedge backtest...")
    portfolio_returns, diagnostics = run_backtest(returns, return_diagnostics=True)

    unhedged = returns[TARGET_ASSET].iloc[-len(portfolio_returns) :]
    metrics = _compute_metrics(unhedged, portfolio_returns)

    print("\n========== PERFORMANCE ==========")
    print("Sharpe Ratio:", round(metrics["sharpe_ratio"], 4))
    print("Sortino Ratio:", round(metrics["sortino_ratio"], 4))
    print("Max Drawdown:", round(metrics["max_drawdown"], 4))
    print("Hedge Effectiveness:", round(metrics["hedge_effectiveness"], 4))

    results_df = pd.DataFrame({"Hedged_Return": portfolio_returns})
    results_df.to_csv("hedged_portfolio_results.csv", index=False)

    if not diagnostics.empty:
        diagnostics.to_csv("hedge_diagnostics.csv", index=False)

    with open("performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to hedged_portfolio_results.csv")
    print("Diagnostics saved to hedge_diagnostics.csv and performance_metrics.json")


if __name__ == "__main__":
    main()
