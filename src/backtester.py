import numpy as np
from config import WINDOW, STEP, DCC_A, DCC_B
from .garch_x_model import fit_garch, fit_garch_x
from .dcc_model import compute_dcc
from .hedge_engine import HedgeEngine
import pandas as pd

def run_backtest(df):

    returns = df.drop(columns=["sentiment_score"])
    sentiment = df["sentiment_score"]
    
    # Initialize HedgeEngine
    engine = HedgeEngine()

    portfolio_returns = []
    
    for t in range(WINDOW, len(df), STEP):
        
        window = returns.iloc[t-WINDOW:t]
        window_sent = sentiment.iloc[t-WINDOW:t].shift(1).fillna(0)
        
        sigmas_last = []
        std_resids = []
        valid_cols = []
        
        for i, col in enumerate(window.columns):
            # Skip columns that are entirely NaN
            if window[col].isna().all():
                continue
            
            valid_cols.append(col)
            
            if col == "NIFTY":
                sigma, resid = fit_garch_x(window[col], window_sent)
            else:
                sigma, resid = fit_garch(window[col])
            
            last_sigma = sigma.iloc[-1]
            # Use a small default if sigma is NaN
            if pd.isna(last_sigma) or np.isinf(last_sigma):
                last_sigma = 0.01
            sigmas_last.append(last_sigma)
            
            # Convert to numpy array and handle NaN
            resid_array = resid.values.flatten()
            resid_array = np.nan_to_num(resid_array, nan=0.0, posinf=0.01, neginf=-0.01)
            std_resids.append(resid_array)
        
        # Skip this window if we don't have valid data
        if len(std_resids) < 1:
            continue
        
        std_resids_array = np.column_stack(std_resids)
        
        # Check for all-zero or all-nan data
        if np.all(std_resids_array == 0) or np.all(np.isnan(std_resids_array)):
            continue
        
        R_series = compute_dcc(std_resids_array, DCC_A, DCC_B)
        R_last = R_series[-1]
        
        H_t = engine.compute_covariance_matrix(R_last, sigmas_last)
        
        hedge_vector = engine.compute_multivariate_hedge(H_t)
        hedge_vector = engine.adjust_for_sentiment(hedge_vector, sentiment.iloc[t])
        
        r_t = returns.iloc[t]
        
        # Only use returns for valid columns (excluding NIFTY which is the portfolio to hedge)
        hedge_assets = [c for c in valid_cols if c != "NIFTY"]
        if len(hedge_assets) == 0:
            # No hedge assets available, just use portfolio return
            portfolio_r = r_t.get("NIFTY", 0)
        else:
            valid_returns = r_t[hedge_assets].values
            valid_returns = np.nan_to_num(valid_returns, nan=0.0, posinf=0.0, neginf=0.0)
            hedge_returns = np.dot(hedge_vector[:len(hedge_assets)], valid_returns)
            portfolio_r = r_t.get("NIFTY", 0) - hedge_returns
        
        portfolio_returns.append(portfolio_r)
    
    return np.array(portfolio_returns) if portfolio_returns else np.array([0])