import pandas as pd
import numpy as np

from src.load_asset import load_all_assets, compute_returns
from src.sentiment_engine import FinBERTSentiment
from src.backtester import run_backtest
from src.evaluation import sharpe_ratio, max_drawdown, hedge_effectiveness


# ==========================================
# 1️⃣ LOAD PRICE DATA
# ==========================================

print("Loading asset prices...")
prices = load_all_assets()

print("Computing returns...")
returns = compute_returns(prices)

# Ensure datetime index
returns.index = pd.to_datetime(returns.index)


# ==========================================
# 2️⃣ LOAD SENTIMENT DATA
# ==========================================

print("Fetching market sentiment...")
sent_engine = FinBERTSentiment()

daily_sentiment = sent_engine.get_daily_sentiment(days=14)

if daily_sentiment.empty:
    print("No sentiment data found. Using zero sentiment.")
    returns["sentiment_score"] = 0.0

else:
    # Convert sentiment date column to datetime index
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    daily_sentiment.set_index("date", inplace=True)

    # Merge on index (clean way)
    returns = returns.join(daily_sentiment, how="left")

    # Fill missing sentiment values
    returns["sentiment_score"] = returns["sentiment_score"].fillna(0.0)


print("Final dataset ready.")
print(returns.tail())


# ==========================================
# 3️⃣ RUN AI-DCC-GARCH-X BACKTEST
# ==========================================

print("\nRunning dynamic hedge backtest...")
portfolio_returns = run_backtest(returns)


# ==========================================
# 4️⃣ EVALUATION
# ==========================================

# Align unhedged NIFTY series
unhedged = returns["NIFTY"].iloc[-len(portfolio_returns):]

print("\n========== PERFORMANCE ==========")
print("Sharpe Ratio:", round(sharpe_ratio(portfolio_returns), 4))
print("Max Drawdown:", round(max_drawdown(portfolio_returns), 4))
print("Hedge Effectiveness:", round(hedge_effectiveness(unhedged, portfolio_returns), 4))


# ==========================================
# 5️⃣ OPTIONAL: SAVE OUTPUT
# ==========================================

results_df = pd.DataFrame({
    "Hedged_Return": portfolio_returns
})

results_df.to_csv("hedged_portfolio_results.csv", index=False)

print("\nResults saved to hedged_portfolio_results.csv")