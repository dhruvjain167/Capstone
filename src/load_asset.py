import os
import pandas as pd
import numpy as np
import yfinance as yf

ASSET_MAP = {
    "NIFTY": "nifty.csv",
    "GOLD": "gold.csv",
    "USDINR": "usdinr.csv",
    "CRUDE": "crude.csv",
    "GSEC10Y": "gsec10y.csv"
}

# Asset tickers for yfinance
ASSET_TICKERS = {
    "NIFTY": "^NSEI",
    "GOLD": "GC=F",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
    "GSEC10Y": "^INDIABOND10Y"
}

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
        print(f"✓ Downloaded {asset_name}: {len(df)} records")
        return df
    except Exception as e:
        print(f"✗ Error downloading {asset_name}: {e}")
        return None

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

def compute_returns(price_df):
    
    # Remove columns with all NaN values
    price_df = price_df.dropna(axis=1, how='all')
    
    # Log returns for price-based assets
    returns = np.log(price_df / price_df.shift(1))
    
    # For yield (GSEC10Y), use first difference instead if it exists
    if "GSEC10Y" in returns.columns and not returns["GSEC10Y"].isna().all():
        returns["GSEC10Y"] = price_df["GSEC10Y"].diff()
    
    # Only drop rows where all values are NaN, not individual NaN values
    return returns.dropna(how='all')

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
        print(f"✓ Assets data saved to assets.csv ({len(merged)} rows)")
        
        # Compute and save returns
        print("\nComputing returns...")
        returns_df = compute_returns(merged)
        returns_df.to_csv("returns.csv")
        print(f"✓ Returns data saved to returns.csv ({len(returns_df)} rows)")
        
        if len(returns_df) > 0:
            print(f"\nReturns statistics:")
            print(returns_df.describe())
    else:
        print("\n✗ Failed to download any asset data")

def align_to_nifty_calendar(df):
    nifty_calendar = df["NIFTY"].dropna().index
    df = df.loc[nifty_calendar]
    return df