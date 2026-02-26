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

# Remove invalid values
returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

# 🔥 Scale returns to percentage (for GARCH stability)
returns = returns * 100


# ==========================================
# 2️⃣ LOAD ASSET-SPECIFIC SENTIMENT
# ==========================================

print("Fetching asset-specific sentiment...")
sent_engine = FinBERTSentiment()

asset_sentiment = sent_engine.get_asset_sentiment(days=14)

if asset_sentiment.empty:
    print("No sentiment data found. Using zero sentiment.")
    sentiment_cols = [
        "NIFTY_sentiment",
        "GOLD_sentiment",
        "USDINR_sentiment",
        "CRUDE_sentiment"
    ]
    for col in sentiment_cols:
        returns[col] = 0.0

else:
    # Ensure datetime index
    asset_sentiment.index = pd.to_datetime(asset_sentiment.index)

    # Merge safely
    returns = returns.join(asset_sentiment, how="left")

    sentiment_cols = [
        "NIFTY_sentiment",
        "GOLD_sentiment",
        "USDINR_sentiment",
        "CRUDE_sentiment"
    ]

    # Ensure all sentiment columns exist
    for col in sentiment_cols:
        if col not in returns.columns:
            returns[col] = 0.0

    # Fill missing sentiment
    returns[sentiment_cols] = returns[sentiment_cols].fillna(0.0)

    # 🔥 VERY IMPORTANT: Lag sentiment (avoid lookahead bias)
    returns[sentiment_cols] = returns[sentiment_cols].shift(1)
    returns[sentiment_cols] = returns[sentiment_cols].fillna(0.0)


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

# Align unhedged returns
unhedged = returns["NIFTY"].iloc[-len(portfolio_returns):]

print("\n========== PERFORMANCE ==========")
print("Sharpe Ratio:", round(sharpe_ratio(portfolio_returns), 4))
print("Max Drawdown:", round(max_drawdown(portfolio_returns), 4))
print("Hedge Effectiveness:", round(hedge_effectiveness(unhedged, portfolio_returns), 4))


# ==========================================
# 5️⃣ SAVE OUTPUT
# ==========================================

results_df = pd.DataFrame({
    "Hedged_Return": portfolio_returns
})

results_df.to_csv("hedged_portfolio_results.csv", index=False)

print("\nResults saved to hedged_portfolio_results.csv")