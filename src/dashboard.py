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
    returns_df["Hedged_Return"].rolling(20).mean() /
    returns_df["Hedged_Return"].rolling(20).std()
)

metrics = {}
if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', np.nan):.4f}")
col2.metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan):.4f}")
col3.metric("Hedge Effectiveness", f"{metrics.get('hedge_effectiveness', np.nan):.4f}")

st.subheader("Equity Curve")
fig_curve = px.line(returns_df, x="Step", y="Cumulative", title="Cumulative Performance of Hedged Portfolio")
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("Return Distribution")
fig_hist = px.histogram(returns_df, x="Hedged_Return", nbins=40, title="Distribution of Hedged Returns")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Rolling Sharpe Ratio (20-step)")
fig_sharpe = px.line(returns_df, x="Step", y="Rolling_Sharpe_20", title="Rolling Sharpe")
st.plotly_chart(fig_sharpe, use_container_width=True)

if diag_path.exists():
    diag_df = pd.read_csv(diag_path)
    if "date" in diag_df.columns:
        diag_df["date"] = pd.to_datetime(diag_df["date"])

    st.subheader("Hedge Dynamics")
    wt_cols = [c for c in diag_df.columns if c.startswith("weight_")]
    if wt_cols:
        melted = diag_df.melt(id_vars=["date"] if "date" in diag_df.columns else None, value_vars=wt_cols,
                              var_name="Asset", value_name="Weight")
        fig_w = px.line(melted, x="date" if "date" in diag_df.columns else melted.index,
                        y="Weight", color="Asset", title="Dynamic Hedge Weights")
        st.plotly_chart(fig_w, use_container_width=True)

    if {"est_vol", "leverage"}.issubset(diag_df.columns):
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=diag_df["date"] if "date" in diag_df.columns else diag_df.index,
                                     y=diag_df["est_vol"], name="Estimated Volatility"))
        fig_vol.add_trace(go.Scatter(x=diag_df["date"] if "date" in diag_df.columns else diag_df.index,
                                     y=diag_df["leverage"], name="Volatility Target Leverage", yaxis="y2"))
        fig_vol.update_layout(
            title="Risk Targeting Controls",
            yaxis=dict(title="EWMA Volatility"),
            yaxis2=dict(title="Leverage", overlaying="y", side="right"),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Parameter Explanations")
st.markdown(
    """
- **DCC_A / DCC_B**: control correlation responsiveness vs persistence. Higher `DCC_A` reacts faster to shocks; high `DCC_B` smooths noise.
- **RIDGE_LAMBDA**: stabilizes covariance inversion to avoid extreme hedge ratios when hedge assets are highly collinear.
- **MAX_HEDGE_WEIGHT**: caps position size per hedge asset and prevents leverage blow-ups.
- **EWMA_LAMBDA**: smoothing for realized volatility estimate used by risk targeting.
- **VOL_TARGET**: target portfolio volatility used to scale exposure and improve Sharpe / drawdown consistency.
- **Sentiment Adjustment**: bearish sentiment slightly scales hedges up, bullish sentiment scales hedges down.
"""
)
