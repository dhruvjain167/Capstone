import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hedging Performance Dashboard", layout="wide")
st.title("AI-DCC-GARCH-X Hedging Dashboard")

metrics_path = Path("performance_metrics.json")
returns_path = Path("hedged_portfolio_results.csv")
diag_path = Path("hedge_diagnostics.csv")

if not returns_path.exists():
    st.error("No backtest output found. Run `python run.py` first.")
    st.stop()

returns_df = pd.read_csv(returns_path)
if "Hedged_Return" not in returns_df.columns:
    st.error("hedged_portfolio_results.csv is missing `Hedged_Return` column.")
    st.stop()

returns_df["Step"] = np.arange(len(returns_df))
returns_df["Cumulative"] = (1 + returns_df["Hedged_Return"]).cumprod()
returns_df["Rolling_Sharpe_20"] = (
    returns_df["Hedged_Return"].rolling(20).mean() / returns_df["Hedged_Return"].rolling(20).std()
)

metrics = {}
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', np.nan):.4f}")
k2.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', np.nan):.4f}")
k3.metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan):.4f}")
k4.metric("Hedge Effectiveness", f"{metrics.get('hedge_effectiveness', np.nan):.4f}")

k5, k6, k7 = st.columns(3)
k5.metric("Ann. Return", f"{metrics.get('annualized_return', np.nan):.2%}")
k6.metric("Ann. Vol", f"{metrics.get('annualized_volatility', np.nan):.2%}")
k7.metric("VaR (95%)", f"{metrics.get('var_95', np.nan):.3%}")

st.subheader("Industry-Standard Interpretation")
interp = []
if metrics:
    sharpe = metrics.get("sharpe_ratio", 0)
    hedge_eff = metrics.get("hedge_effectiveness", 0)
    mdd = metrics.get("max_drawdown", 0)

    interp.append(
        f"Sharpe {sharpe:.3f}: {'healthy risk-adjusted performance' if sharpe > 0.5 else 'strategy still has weak risk-adjusted returns; improve signal quality and hedge timing'}"
    )
    interp.append(
        f"Hedge effectiveness {hedge_eff:.3f}: {'variance is reduced vs unhedged benchmark' if hedge_eff > 0 else 'hedge currently increases variance; focus on ratio stability and correlation filtering'}"
    )
    interp.append(
        f"Max drawdown {mdd:.3f}: {'drawdown is controlled' if abs(mdd) < 0.12 else 'tail risk remains high; reduce gross hedge and turnover during unstable regimes'}"
    )

st.markdown("\n".join([f"- {x}" for x in interp]) if interp else "Run backtest to generate interpretation.")

st.subheader("Equity Curve")
fig_curve = px.line(returns_df, x="Step", y="Cumulative", title="Cumulative Performance of Hedged Portfolio")
st.plotly_chart(fig_curve, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Return Distribution")
    fig_hist = px.histogram(returns_df, x="Hedged_Return", nbins=40, title="Distribution of Hedged Returns")
    st.plotly_chart(fig_hist, use_container_width=True)
with c2:
    st.subheader("Rolling Sharpe Ratio (20-step)")
    fig_sharpe = px.line(returns_df, x="Step", y="Rolling_Sharpe_20", title="Rolling Sharpe")
    st.plotly_chart(fig_sharpe, use_container_width=True)

if diag_path.exists():
    diag_df = pd.read_csv(diag_path)
    if "date" in diag_df.columns:
        diag_df["date"] = pd.to_datetime(diag_df["date"])

    st.subheader("Hedge Ratio Dynamics")
    final_weight_cols = [c for c in diag_df.columns if c.startswith("weight_final_")]
    if final_weight_cols:
        melted = diag_df.melt(
            id_vars=["date"] if "date" in diag_df.columns else None,
            value_vars=final_weight_cols,
            var_name="Asset",
            value_name="Hedge Ratio",
        )
        melted["Asset"] = melted["Asset"].str.replace("weight_final_", "", regex=False)
        fig_w = px.line(
            melted,
            x="date" if "date" in diag_df.columns else melted.index,
            y="Hedge Ratio",
            color="Asset",
            title="Final Hedge Ratios by Asset",
        )
        st.plotly_chart(fig_w, use_container_width=True)

    st.subheader("Asset Allocation from Hedge Ratios")
    alloc_cols = [c for c in diag_df.columns if c.startswith("alloc_")]
    if alloc_cols:
        alloc_assets = [c.replace("alloc_", "") for c in alloc_cols]
        avg_alloc = diag_df[alloc_cols].mean().values
        latest_alloc = diag_df[alloc_cols].iloc[-1].values

        alloc_summary = pd.DataFrame(
            {
                "Asset": alloc_assets,
                "Average Allocation": avg_alloc,
                "Latest Allocation": latest_alloc,
            }
        )
        st.dataframe(alloc_summary.style.format({"Average Allocation": "{:.2%}", "Latest Allocation": "{:.2%}"}))

        p1, p2 = st.columns(2)
        with p1:
            fig_avg = px.pie(alloc_summary, names="Asset", values="Average Allocation", title="Average Hedge Allocation")
            st.plotly_chart(fig_avg, use_container_width=True)
        with p2:
            fig_latest = px.bar(alloc_summary, x="Asset", y="Latest Allocation", title="Latest Hedge Allocation")
            st.plotly_chart(fig_latest, use_container_width=True)

    st.subheader("Sentiment Impact on Hedge Ratios")
    if {"sentiment_score", "sentiment_multiplier"}.issubset(diag_df.columns):
        s1, s2 = st.columns(2)
        with s1:
            fig_scatter = px.scatter(
                diag_df,
                x="sentiment_score",
                y="sentiment_multiplier",
                title="Sentiment Score vs Hedge Multiplier",
                trendline="ols",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with s2:
            raw_cols = [c for c in diag_df.columns if c.startswith("weight_raw_")]
            final_cols = [c for c in diag_df.columns if c.startswith("weight_final_")]
            if raw_cols and final_cols:
                diag_df["gross_raw"] = diag_df[raw_cols].abs().sum(axis=1)
                diag_df["gross_final"] = diag_df[final_cols].abs().sum(axis=1)
                fig_gross = go.Figure()
                x_axis = diag_df["date"] if "date" in diag_df.columns else diag_df.index
                fig_gross.add_trace(go.Scatter(x=x_axis, y=diag_df["gross_raw"], name="Raw Gross Hedge"))
                fig_gross.add_trace(go.Scatter(x=x_axis, y=diag_df["gross_final"], name="Final Gross Hedge"))
                fig_gross.update_layout(title="Effect of Sentiment + Smoothing on Gross Hedge Exposure")
                st.plotly_chart(fig_gross, use_container_width=True)

        st.markdown(
            """
**How sentiment changes hedge ratios:**
- The model maps sentiment score into a smooth multiplier: **`multiplier = 1 - strength * tanh(sentiment)`**.
- Positive sentiment (risk-on) produces multiplier < 1 and reduces hedge notional.
- Negative sentiment (risk-off) produces multiplier > 1 and increases hedge notional.
- Final weights are then smoothed and gross-capped, so sentiment affects ratios but does not cause extreme jumps.
"""
        )

    if {"est_vol", "leverage", "turnover", "transaction_cost"}.issubset(diag_df.columns):
        st.subheader("Risk and Implementation Controls")
        fig_risk = go.Figure()
        x_axis = diag_df["date"] if "date" in diag_df.columns else diag_df.index
        fig_risk.add_trace(go.Scatter(x=x_axis, y=diag_df["est_vol"], name="Estimated Volatility"))
        fig_risk.add_trace(go.Scatter(x=x_axis, y=diag_df["leverage"], name="Vol Target Leverage", yaxis="y2"))
        fig_risk.update_layout(
            title="EWMA Volatility Targeting",
            yaxis=dict(title="Estimated Volatility"),
            yaxis2=dict(title="Leverage", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        fig_impl = px.line(
            diag_df,
            x="date" if "date" in diag_df.columns else diag_df.index,
            y=["turnover", "transaction_cost"],
            title="Turnover and Transaction Cost Proxy",
        )
        st.plotly_chart(fig_impl, use_container_width=True)

st.subheader("Parameter Explanations")
st.markdown(
    """
- **DCC_A / DCC_B**: responsiveness vs persistence of dynamic correlations.
- **RIDGE_LAMBDA**: regularization for hedge-covariance inversion (improves numerical stability).
- **MAX_HEDGE_WEIGHT / MAX_GROSS_EXPOSURE**: caps single-asset and total hedge notional.
- **WEIGHT_SMOOTHING_ALPHA**: reduces hedge-ratio whipsaw and turnover.
- **MIN_CORRELATION_CONFIDENCE**: de-risks hedge when target-hedge correlation weakens.
- **SENTIMENT_STRENGTH**: controls how strongly news sentiment scales hedge ratios.
- **VOL_TARGET / EWMA_LAMBDA**: risk-targeting to stabilize realized volatility.
- **TRANSACTION_COST_BPS**: implementation drag proxy to avoid overfitting unrealistic gross returns.
"""
)
