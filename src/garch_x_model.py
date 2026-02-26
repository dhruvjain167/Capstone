import numpy as np
import pandas as pd
from arch import arch_model

# ==========================================
# STANDARD GARCH
# ==========================================

def fit_garch(series):
    series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(series_clean) < 5:
        std_val = series_clean.std() if len(series_clean) > 0 else 0.01
        return pd.Series(std_val, index=series.index), pd.Series(0.0, index=series.index)

    # FIX: Add rescale=True here to let arch handle the small values automatically
    model = arch_model(
        series_clean,
        mean="Constant",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        rescale=True  # This tells the model to scale data to ~1.0 variance internally
    )

    # FIX: Remove any keyword arguments from fit() that might cause TypeErrors
    res = model.fit(disp="off", show_warning=False)

    # If the model rescaled the data, we need to scale the volatility back down
    scale = res.scale if hasattr(res, 'scale') else 1.0
    sigma = res.conditional_volatility / scale
    std_resid = res.resid / res.conditional_volatility

    sigma_full = pd.Series(0.01, index=series.index)
    sigma_full.loc[series_clean.index] = sigma

    std_resid_full = pd.Series(0.0, index=series.index)
    std_resid_full.loc[series_clean.index] = std_resid

    return sigma_full, std_resid_full


# ==========================================
# GARCH-X (With Sentiment Exogenous Variable)
# ==========================================

def fit_garch_x(series, exog):
    series_clean = series.replace([np.inf, -np.inf], np.nan)
    exog_clean = exog.replace([np.inf, -np.inf], np.nan)

    mask = ~(series_clean.isna() | exog_clean.isna())
    series_clean = series_clean[mask]
    exog_clean = exog_clean[mask]

    if len(series_clean) < 5:
        std_val = series_clean.std() if len(series_clean) > 0 else 0.01
        return pd.Series(std_val, index=series.index), pd.Series(0.0, index=series.index)

    # FIX: Set rescale=True and pass exogenous variables properly
    model = arch_model(
        series_clean,
        x=exog_clean, # Note: x in the constructor is for the mean equation
        mean="Constant",
        vol="Garch",
        p=1,
        q=1,
        dist="normal",
        rescale=True 
    )

    res = model.fit(disp="off", show_warning=False)

    # Adjustment for rescaling
    scale = res.scale if hasattr(res, 'scale') else 1.0
    sigma = res.conditional_volatility / scale
    std_resid = res.resid / res.conditional_volatility

    sigma_full = pd.Series(0.01, index=series.index)
    sigma_full.loc[series_clean.index] = sigma

    std_resid_full = pd.Series(0.0, index=series.index)
    std_resid_full.loc[series_clean.index] = std_resid

    return sigma_full, std_resid_full