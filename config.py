WINDOW = 20  # Trading days for volatility window (1 month)
STEP = 5

ASSETS = ["NIFTY", "GOLD", "USDINR", "CRUDE", "GSEC10Y"]

TARGET_ASSET = "NIFTY"
HEDGE_ASSETS = ["GOLD", "USDINR", "CRUDE", "GSEC10Y"]

DCC_A = 0.02
DCC_B = 0.95

# Risk and hedge controls
RIDGE_LAMBDA = 1e-4
MAX_HEDGE_WEIGHT = 1.25
VOL_TARGET = 0.008
EWMA_LAMBDA = 0.94
