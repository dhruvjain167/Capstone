import os
import pandas as pd
import numpy as np
import yfinance as yf

ASSET_MAP = {
    "NIFTY": "nifty.csv",
    "NIFTY_FUT": "nifty_fut.csv",
    "GOLD": "gold.csv",
    "USDINR": "usdinr.csv",
    "CRUDE": "crude.csv",
    "GSEC10Y": "gsec10y.csv"
}

# Asset tickers for yfinance
ASSET_TICKERS = {
    "NIFTY": "^NSEI",
    "NIFTY_FUT": "^NSEI",  # Nifty Futures — use Nifty index as proxy; real ticker: "NIFTY_FUT.NS" or specific contract
    "GOLD": "GC=F",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
    "GSEC10Y": "^INDIABOND10Y"
}

# Monthly GSEC10Y data embedded for reliable fallback (yields, not prices)
GSEC10Y_MONTHLY_FALLBACK = True


def load_single_asset(path, col_name):
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df[['Date', 'Close']]
    df.columns = ['Date', col_name]
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df


def download_asset_from_yfinance(ticker, asset_name, start_date="2020-01-01", end_date=None):
    """Download asset data from yfinance"""
    print(f"Downloading {asset_name} ({ticker})...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]
        df.columns = ['Date', asset_name]
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        print(f"[OK] Downloaded {asset_name}: {len(df)} records")
        return df
    except Exception as e:
        print(f"[ERROR] Error downloading {asset_name}: {e}")
        return None


def _load_gsec10y_monthly(project_root=None):
    """Load and interpolate monthly GSEC10Y yield data to daily frequency."""
    search_paths = [
        "gsec10y.csv",
        os.path.join(os.path.dirname(__file__), "..", "gsec10y.csv"),
    ]
    if project_root:
        search_paths.insert(0, os.path.join(project_root, "gsec10y.csv"))

    found = None
    for p in search_paths:
        if os.path.exists(p):
            found = p
            break

    if found is None:
        print("Warning: gsec10y.csv not found for monthly yield interpolation.")
        return None

    df = pd.read_csv(found, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df = df.sort_index()
    df.columns = ["GSEC10Y"]

    # Resample monthly to daily with cubic interpolation for smooth yield curve
    df_daily = df.resample("D").interpolate(method="cubic")
    print(f"[OK] GSEC10Y: Monthly yields interpolated to {len(df_daily)} daily records")
    return df_daily


def load_all_assets(data_path="data/raw/"):
    # If a pre-merged assets file exists, prefer that for convenience.
    possible_merged = [
        "assets.csv",
        os.path.join(os.path.dirname(__file__), "assets.csv")
    ]

    for merged_path in possible_merged:
        if os.path.exists(merged_path):
            df = pd.read_csv(merged_path, parse_dates=["Date"]) 
            df.set_index('Date', inplace=True)
            df = df.sort_index()
            
            # If NIFTY_FUT column doesn't exist, create a synthetic one from NIFTY
            if "NIFTY_FUT" not in df.columns and "NIFTY" in df.columns:
                print("Info: Creating synthetic NIFTY_FUT from NIFTY (futures ~ spot + carry).")
                # Nifty Futures typically trade at a small premium (carry) to spot
                # We approximate using 0.05% daily premium noise to simulate basis
                np.random.seed(42)
                basis_noise = np.random.normal(1.0002, 0.0003, len(df))
                df["NIFTY_FUT"] = df["NIFTY"] * basis_noise

            # If GSEC10Y is all NaN or missing, attempt monthly interpolation
            if "GSEC10Y" not in df.columns or df["GSEC10Y"].isna().all():
                gsec = _load_gsec10y_monthly()
                if gsec is not None:
                    df = df.join(gsec, how="left", rsuffix="_monthly")
                    if "GSEC10Y_monthly" in df.columns:
                        df["GSEC10Y"] = df["GSEC10Y_monthly"]
                        df.drop(columns=["GSEC10Y_monthly"], inplace=True)
                    df["GSEC10Y"] = df["GSEC10Y"].interpolate(method="linear")
                    df["GSEC10Y"] = df["GSEC10Y"].ffill().bfill()

            return df

    dfs = []
    missing = []

    for asset, file in ASSET_MAP.items():
        # search common locations for the CSV
        candidates = [
            os.path.join(data_path, file),
            os.path.join(os.path.dirname(__file__), file),
            os.path.join(os.getcwd(), file),
            file
        ]

        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break

        if not found:
            missing.append(file)
            continue

        df = load_single_asset(found, asset)
        dfs.append(df)

    # If NIFTY is missing we cannot proceed
    loaded_assets = [d.columns[0] for d in dfs]
    if "NIFTY" not in loaded_assets:
        raise FileNotFoundError(
            f"NIFTY data is required but not found. Searched locations for: {missing}"
        )

    # For any missing non-critical assets, create NaN columns aligned to NIFTY index
    if missing:
        print(f"Warning: missing asset files: {missing}. Filling missing columns with NaN aligned to NIFTY.")
        nifty_df = None
        for d in dfs:
            if "NIFTY" in d.columns:
                nifty_df = d
                break

        for m in missing:
            if m in ASSET_MAP.values():
                # map filename back to asset key
                asset_key = None
                for k, v in ASSET_MAP.items():
                    if v == m:
                        asset_key = k
                        break
                if asset_key is None:
                    continue

                # create empty series indexed by NIFTY dates
                empty_df = pd.DataFrame(index=nifty_df.index)
                empty_df[asset_key] = np.nan
                dfs.append(empty_df)

    merged = pd.concat(dfs, axis=1)
    merged = merged.sort_index()

    # Forward fill small gaps
    merged = merged.ffill()

    return merged


def _fix_fake_zeros(returns_df, threshold=0.0):
    """
    CRITICAL FIX: Replace 0.0 values that represent missing data, not real returns.
    
    0.0 returns occur when:
    - Market was closed (holiday/weekend) but data was forward-filled in prices
    - Data source returned no value and defaulted to 0
    
    Strategy: Replace exact 0.0 returns with NaN, then forward-fill.
    We preserve returns that are very close to but not exactly zero.
    """
    # Create mask for exact zeros across all asset columns (not sentiment or other derived cols)
    asset_cols = [c for c in returns_df.columns if c not in {"sentiment_score"}]
    
    for col in asset_cols:
        # Count zeros before fix
        zero_count = (returns_df[col] == threshold).sum()
        if zero_count > 0:
            # Replace exact 0.0 with NaN
            returns_df[col] = returns_df[col].replace(threshold, np.nan)
            print(f"  Fixed {zero_count} fake zero returns in {col}")
    
    return returns_df


def compute_returns(price_df):
    
    # Remove columns with all NaN values
    price_df = price_df.dropna(axis=1, how='all')
    
    # Log returns for price-based assets
    returns = np.log(price_df / price_df.shift(1))
    
    # For yield (GSEC10Y), use change in yield (first difference) instead of log returns
    # This converts yield levels to yield changes, which better captures bond risk dynamics
    if "GSEC10Y" in returns.columns and not returns["GSEC10Y"].isna().all():
        returns["GSEC10Y"] = price_df["GSEC10Y"].diff()
    
    # ============================================
    # CRITICAL FIX 1: Replace fake zero returns
    # ============================================
    print("Preprocessing: Fixing fake zero returns...")
    returns = _fix_fake_zeros(returns)
    
    # ============================================
    # CRITICAL FIX 2: Handle missing values properly  
    # DO NOT leave NaNs or treat them as zero!
    # Forward-fill first, then back-fill remaining edge NaNs
    # ============================================
    print("Preprocessing: Forward-filling missing values...")
    returns = returns.ffill()
    returns = returns.bfill()  # Only for initial NaNs at the start
    
    # Only drop rows where all values are NaN (edge case after processing)
    returns = returns.dropna(how='all')
    
    # Final sanity check: report any remaining issues
    remaining_nans = returns.isna().sum().sum()
    if remaining_nans > 0:
        print(f"Warning: {remaining_nans} NaN values remain after preprocessing")
    else:
        print("[OK] Data preprocessing complete: no NaN or fake zeros remain")
    
    return returns


if __name__ == "__main__":
    # Download data from yfinance
    print("=" * 50)
    print("Downloading asset data from yfinance...")
    print("=" * 50)
    
    dfs = []
    for asset, ticker in ASSET_TICKERS.items():
        df = download_asset_from_yfinance(ticker, asset)
        if df is not None:
            dfs.append(df)
    
    if dfs:
        # Merge all assets
        print("\nMerging assets...")
        merged = pd.concat(dfs, axis=1)
        merged = merged.sort_index()
        
        # Forward fill then backward fill to handle missing values
        merged = merged.ffill().bfill()
        
        # Save merged assets
        merged.to_csv("assets.csv")
        print(f"[OK] Assets data saved to assets.csv ({len(merged)} rows)")
        
        # Compute and save returns
        print("\nComputing returns...")
        returns_df = compute_returns(merged)
        returns_df.to_csv("returns.csv")
        print(f"[OK] Returns data saved to returns.csv ({len(returns_df)} rows)")
        
        if len(returns_df) > 0:
            print(f"\nReturns statistics:")
            print(returns_df.describe())
    else:
        print("\n[ERROR] Failed to download any asset data")

def align_to_nifty_calendar(df):
    nifty_calendar = df["NIFTY"].dropna().index
    df = df.loc[nifty_calendar]
    return df