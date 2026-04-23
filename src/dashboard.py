import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Hedging Performance Dashboard", layout="wide")
st.title("🛡️ AI-DCC-GARCH-X Hedging Dashboard")

metrics_path = Path("performance_metrics.json")
returns_path = Path("hedged_portfolio_results.csv")
diag_path = Path("hedge_diagnostics.csv")
raw_returns_path = Path("returns.csv")

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

# ── KPI Cards ──
col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', np.nan):.4f}")
col2.metric("Max Drawdown", f"{metrics.get('max_drawdown', np.nan):.4f}")
col3.metric("Hedge Effectiveness", f"{metrics.get('hedge_effectiveness', np.nan):.4f}")

st.subheader("📈 Equity Curve")
fig_curve = px.line(returns_df, x="Step", y="Cumulative", title="Cumulative Performance of Hedged Portfolio")
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("📊 Return Distribution")
fig_hist = px.histogram(returns_df, x="Hedged_Return", nbins=40, title="Distribution of Hedged Returns")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("📉 Rolling Sharpe Ratio (20-step)")
fig_sharpe = px.line(returns_df, x="Step", y="Rolling_Sharpe_20", title="Rolling Sharpe")
st.plotly_chart(fig_sharpe, use_container_width=True)

# ── Load diagnostics for advanced plots ──
if diag_path.exists():
    diag_df = pd.read_csv(diag_path)
    if "date" in diag_df.columns:
        diag_df["date"] = pd.to_datetime(diag_df["date"])

    # ════════════════════════════════════════════
    # 1. ROLLING HEDGE RATIOS PLOT
    # ════════════════════════════════════════════
    st.subheader("⚖️ Rolling Hedge Ratios")
    wt_cols = [c for c in diag_df.columns if c.startswith("weight_")]
    if wt_cols:
        x_col = "date" if "date" in diag_df.columns else diag_df.index
        melted = diag_df.melt(
            id_vars=["date"] if "date" in diag_df.columns else None,
            value_vars=wt_cols, var_name="Asset", value_name="Weight"
        )
        melted["Asset"] = melted["Asset"].str.replace("weight_", "")
        fig_w = px.line(melted, x="date" if "date" in diag_df.columns else melted.index,
                        y="Weight", color="Asset", title="Dynamic Hedge Weights Over Time")
        fig_w.update_layout(hovermode="x unified")
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("No hedge weight columns found in diagnostics.")

    # ════════════════════════════════════════════
    # 2. CORRELATION HEATMAP
    # ════════════════════════════════════════════
    st.subheader("🔥 Correlation Heatmap")
    corr_cols = [c for c in diag_df.columns if c.startswith("corr_")]
    if corr_cols:
        # Show latest window correlation
        latest_corrs = diag_df[corr_cols].iloc[-1] if len(diag_df) > 0 else pd.Series()
        corr_labels = [c.replace("corr_", "") for c in corr_cols]

        # Build full correlation matrix from raw returns if available
        if raw_returns_path.exists():
            raw_df = pd.read_csv(raw_returns_path, index_col=0, parse_dates=True)
            asset_cols = [c for c in raw_df.columns if c not in {"sentiment_score"}]
            corr_matrix = raw_df[asset_cols].corr()
            fig_hm = px.imshow(
                corr_matrix, text_auto=".2f",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Asset Return Correlations (Full Sample)"
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            # Fallback: show rolling correlations as a bar chart
            fig_bar = px.bar(x=corr_labels, y=latest_corrs.values,
                             title="Latest Rolling Correlations with NIFTY",
                             labels={"x": "Asset", "y": "Correlation"})
            st.plotly_chart(fig_bar, use_container_width=True)

        # Rolling correlation time series
        st.subheader("📐 Rolling Correlations Over Time")
        corr_melted = diag_df.melt(
            id_vars=["date"] if "date" in diag_df.columns else None,
            value_vars=corr_cols, var_name="Pair", value_name="Correlation"
        )
        corr_melted["Pair"] = corr_melted["Pair"].str.replace("corr_", "NIFTY↔")
        fig_rc = px.line(corr_melted, x="date" if "date" in diag_df.columns else corr_melted.index,
                         y="Correlation", color="Pair", title="Rolling Pairwise Correlations")
        fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("No correlation data in diagnostics. Re-run backtest to generate.")

    # ════════════════════════════════════════════
    # 3. REGIME CLASSIFICATION VISUALIZATION
    # ════════════════════════════════════════════
    if "regime" in diag_df.columns and "date" in diag_df.columns:
        st.subheader("🌡️ Regime Classification")

        regime_colors = {"crisis": "red", "high_vol": "orange", "normal": "green", "low_vol": "blue"}
        regime_map = {"crisis": 0, "high_vol": 1, "normal": 2, "low_vol": 3}
        diag_df["regime_num"] = diag_df["regime"].map(regime_map)

        fig_regime = go.Figure()
        for regime_name, color in regime_colors.items():
            mask = diag_df["regime"] == regime_name
            if mask.any():
                fig_regime.add_trace(go.Scatter(
                    x=diag_df.loc[mask, "date"], y=diag_df.loc[mask, "est_vol"],
                    mode="markers", name=regime_name.replace("_", " ").title(),
                    marker=dict(color=color, size=6, opacity=0.7)
                ))
        fig_regime.update_layout(title="Market Regimes by Volatility", yaxis_title="Est. Volatility",
                                 xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_regime, use_container_width=True)

        # Regime distribution pie
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            regime_counts = diag_df["regime"].value_counts()
            fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index,
                             title="Regime Distribution", color=regime_counts.index,
                             color_discrete_map=regime_colors)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_r2:
            # Avg return by regime
            if "hedged_return" in diag_df.columns:
                regime_stats = diag_df.groupby("regime")["hedged_return"].agg(["mean", "std", "count"])
                regime_stats.columns = ["Mean Return", "Std Dev", "Count"]
                regime_stats["Sharpe (approx)"] = regime_stats["Mean Return"] / regime_stats["Std Dev"].replace(0, np.nan)
                st.dataframe(regime_stats.round(6), use_container_width=True)

    # ════════════════════════════════════════════
    # 4. HEDGED vs UNHEDGED RETURNS
    # ════════════════════════════════════════════
    st.subheader("🆚 Hedged vs Unhedged Returns")
    if "target_return" in diag_df.columns and "hedged_return" in diag_df.columns:
        x_axis = diag_df["date"] if "date" in diag_df.columns else diag_df.index
        cum_unhedged = (1 + diag_df["target_return"]).cumprod()
        cum_hedged = (1 + diag_df["hedged_return"]).cumprod()

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=x_axis, y=cum_unhedged, name="Unhedged (NIFTY)", line=dict(color="red", width=2)))
        fig_vs.add_trace(go.Scatter(x=x_axis, y=cum_hedged, name="Hedged Portfolio", line=dict(color="green", width=2)))
        fig_vs.update_layout(title="Cumulative: Hedged vs Unhedged", yaxis_title="Growth of ₹1",
                             xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_vs, use_container_width=True)

        # Return scatter
        fig_scatter = px.scatter(diag_df, x="target_return", y="hedged_return",
                                 title="Hedged vs Unhedged Daily Returns",
                                 labels={"target_return": "Unhedged Return", "hedged_return": "Hedged Return"},
                                 opacity=0.5)
        fig_scatter.add_shape(type="line", x0=-0.1, y0=-0.1, x1=0.1, y1=0.1,
                              line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ════════════════════════════════════════════
    # 5. HEDGE EFFECTIVENESS OVER TIME
    # ════════════════════════════════════════════
    st.subheader("📏 Hedge Effectiveness Over Time")
    if "target_return" in diag_df.columns and "hedged_return" in diag_df.columns:
        roll_window = st.slider("Rolling window for effectiveness", 10, 60, 20, key="he_window")

        rolling_var_unhedged = diag_df["target_return"].rolling(roll_window).var()
        rolling_var_hedged = diag_df["hedged_return"].rolling(roll_window).var()
        rolling_he = 1 - (rolling_var_hedged / rolling_var_unhedged.replace(0, np.nan))

        fig_he = go.Figure()
        fig_he.add_trace(go.Scatter(
            x=diag_df["date"] if "date" in diag_df.columns else diag_df.index,
            y=rolling_he, name=f"Rolling HE ({roll_window}-step)",
            line=dict(color="purple", width=2)
        ))
        fig_he.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No benefit")
        fig_he.add_hline(y=0.5, line_dash="dot", line_color="orange", annotation_text="50% effective")
        fig_he.update_layout(title=f"Rolling Hedge Effectiveness ({roll_window}-step window)",
                             yaxis_title="Hedge Effectiveness (1 - Var_hedged/Var_unhedged)",
                             xaxis_title="Date")
        st.plotly_chart(fig_he, use_container_width=True)

    # ── Risk Targeting Controls ──
    if {"est_vol", "leverage"}.issubset(diag_df.columns):
        st.subheader("🎯 Risk Targeting Controls")
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

# ── Parameter Explanations ──
st.subheader("📖 Parameter Explanations")
st.markdown(
    """
- **DCC_A / DCC_B**: control correlation responsiveness vs persistence. Higher `DCC_A` reacts faster to shocks; high `DCC_B` smooths noise.
- **RIDGE_LAMBDA**: stabilizes covariance inversion to avoid extreme hedge ratios when hedge assets are highly collinear.
- **MAX_HEDGE_WEIGHT**: caps position size per hedge asset and prevents leverage blow-ups.
- **EWMA_LAMBDA**: smoothing for realized volatility estimate used by risk targeting.
- **VOL_TARGET**: target portfolio volatility used to scale exposure and improve Sharpe / drawdown consistency.
- **Sentiment Adjustment**: bearish sentiment slightly scales hedges up, bullish sentiment scales hedges down.
- **Regime Classification**: crisis (high vol + negative momentum), high_vol, normal, low_vol — used for adaptive exposure.
"""
)
