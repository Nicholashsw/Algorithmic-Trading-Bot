"""
╔══════════════════════════════════════════════════════════════════════╗
║        VRP STRATEGY — MASTER FILE (All-in-One)                      ║
║        Volatility Risk Premium · Multi-Asset FX · Options           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Author : Nicholas Hong | nhong001@e.ntu.edu.sg                      ║
║  Mentor : Jirong Huang, Eastspring Investments                       ║
║  GitHub : Nicholashsw/Algorithmic-Trading-Bot                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Consolidates (replaces) all scattered files:                        ║
║    strategies/mean_reversion.py                                      ║
║    utils/indicators.py · utils/vrp_signal.py                        ║
║    utils/backtest_engine.py                                          ║
║    options/black_scholes.py                                          ║
║    engine/options_backtest.py · engine/options_signal_runner.py      ║
║    engine/portfolio.py · engine/backtest.py                          ║
║    run_backtest.py · optimize.py                                     ║
║    fetchers/fetch_data.py · fetchers/ibkr_fetch.py                   ║
║    fetchers/fetch_6e_continuous.py · fetchers/fetch_6e_proxy.py      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Datasets (data/ folder):                                            ║
║    EURUSD_yf.csv  · USDJPY_yf.csv  · GBPUSD_yf.csv · AUDUSD_yf.csv ║
║    EURUSD_IBKR.csv · 6E_proxy.csv · 6E.csv                          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Usage:                                                              ║
║    python vrp_master.py              → run full backtest pipeline    ║
║    python vrp_master.py --optimize   → run parameter grid search     ║
║    python vrp_master.py --ibkr       → fetch live data via IBKR      ║
║    import vrp_master as vrp          → use in notebooks              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import warnings
import argparse
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.stats import norm
from scipy.optimize import brentq

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════════════════════

ASSETS = {
    "EURUSD": {"file": "data/EURUSD_yf.csv",  "ticker": "EURUSD=X", "etf": "FXE",  "vrp_premium": 0.020},
    "USDJPY": {"file": "data/USDJPY_yf.csv",  "ticker": "USDJPY=X", "etf": "FXY",  "vrp_premium": 0.015},
    "GBPUSD": {"file": "data/GBPUSD_yf.csv",  "ticker": "GBPUSD=X", "etf": "FXB",  "vrp_premium": 0.020},
    "AUDUSD": {"file": "data/AUDUSD_yf.csv",  "ticker": "AUDUSD=X", "etf": "FXA",  "vrp_premium": 0.025},
}

# Optimal parameters from walk-forward grid search (March 2026)
OPTIMAL_PARAMS = {
    "EURUSD": {"window": 10, "num_std": 1.5, "regime_mode": "none"},
    "USDJPY": {"window": 20, "num_std": 2.0, "regime_mode": "trend_aware"},
    "GBPUSD": {"window": 10, "num_std": 1.5, "regime_mode": "trend_aware"},
    "AUDUSD": {"window": 15, "num_std": 1.5, "regime_mode": "trend_aware"},
}

BACKTEST_CFG = {
    "transaction_cost_pct": 0.0001,   # 1 pip per trade
    "initial_capital":      100.0,
}

OPTIONS_CFG = {
    "r":                  0.025,   # risk-free rate (~2024 level)
    "T":                  1 / 12,  # 1-month to expiry
    "otm_pct":            0.01,    # 1% OTM spread legs
    "use_dynamic_sigma":  True,
    "apply_vrp_filter":   True,
    "holding_bars":       21,      # ~1 month of trading days
}

PARAM_GRID = {
    "window":  [10, 15, 20, 25, 30],
    "num_std": [1.5, 2.0, 2.5],
    "regime":  ["none", "trend_aware"],
}

COLORS = {
    "EURUSD": "#1f77b4",
    "USDJPY": "#ff7f0e",
    "GBPUSD": "#2ca02c",
    "AUDUSD": "#d62728",
    "PORTFOLIO": "#333333",
}


# ════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING & FETCHING
# ════════════════════════════════════════════════════════════════

def load_asset(name: str, file_path: str = None) -> pd.DataFrame:
    """
    Load FX price data from CSV.  Falls back to yfinance if file missing.

    Parameters
    ----------
    name      : Asset key, e.g. 'EURUSD'
    file_path : Path to CSV file (auto-detected from ASSETS config if None)

    Returns
    -------
    DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    path = file_path or ASSETS.get(name, {}).get("file", f"data/{name}_yf.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close"]].dropna(subset=["close"])
        df = df[df["close"] > 0].sort_index()
        print(f"  [{name}] Loaded {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
        return df

    # Fallback: fetch via yfinance
    if YFINANCE_AVAILABLE:
        ticker = ASSETS.get(name, {}).get("ticker", f"{name}=X")
        print(f"  [{name}] File not found. Fetching from yfinance ({ticker})...")
        raw = yf.download(ticker, start="2004-01-01", auto_adjust=True, progress=False)
        if not raw.empty:
            raw.columns = [c.lower() for c in raw.columns]
            raw.index.name = "date"
            raw = raw[["open", "high", "low", "close"]].dropna()
            os.makedirs("data", exist_ok=True)
            raw.to_csv(path)
            print(f"  [{name}] Saved to {path}")
            return raw

    raise FileNotFoundError(f"No data for {name}. Provide {path} or install yfinance.")


def load_all_assets() -> dict:
    """Load all configured assets. Returns dict of {name: DataFrame}."""
    results = {}
    for name in ASSETS:
        try:
            results[name] = load_asset(name)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
    return results


def fetch_yfinance(ticker: str, start: str = "2004-01-01", end: str = None, save_path: str = None) -> pd.DataFrame:
    """
    Generic yfinance downloader.

    Parameters
    ----------
    ticker    : Yahoo Finance ticker, e.g. 'EURUSD=X'
    start     : Start date string 'YYYY-MM-DD'
    end       : End date string (defaults to today)
    save_path : Optional path to save CSV

    Returns
    -------
    DataFrame with OHLCV columns
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("pip install yfinance")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path)
    return df


def fetch_ibkr_spot(symbol: str = "EURUSD", duration: str = "20 Y",
                    port: int = 7497, client_id: int = 1, save: bool = True):
    """
    Fetch spot FX data from Interactive Brokers via ib_insync.

    Requires: TWS or IB Gateway running on localhost:{port}
    Paper trading port: 7497 | Live port: 7496

    Parameters
    ----------
    symbol    : FX pair, e.g. 'EURUSD'
    duration  : IBKR duration string, e.g. '20 Y'
    port      : TWS/Gateway port
    client_id : IBKR client ID (avoid conflicts with TWS default=0)
    save      : If True, save to data/{symbol}_IBKR.csv

    Returns
    -------
    DataFrame of historical bars
    """
    try:
        from ib_insync import IB, Forex, util
    except ImportError:
        raise ImportError("pip install ib_insync")

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id)
    contract = Forex(symbol)
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=False,
        formatDate=1,
    )
    df = util.df(bars)
    if save and not df.empty:
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/{symbol}_IBKR.csv", index=False)
        print(f"  Saved data/{symbol}_IBKR.csv ({len(df)} rows)")
    ib.disconnect()
    return df


def fetch_ibkr_6e_continuous(port: int = 7497, client_id: int = 10, save: bool = True):
    """
    Fetch 6E (Euro FX Futures) continuous contract from IBKR.
    Iterates quarterly expiries (Mar/Jun/Sep/Dec) back to 2004.

    Requires: TWS or IB Gateway running locally.
    """
    try:
        from ib_insync import IB, Future, util
    except ImportError:
        raise ImportError("pip install ib_insync")

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id)
    os.makedirs("data", exist_ok=True)

    years  = list(range(2004, datetime.now().year + 1))
    months = ["03", "06", "09", "12"]
    expiries = [f"{y}{m}" for y in years for m in months]

    all_dfs = []
    for expiry in expiries:
        try:
            contract = Future(symbol="6E", lastTradeDateOrContractMonth=expiry,
                              exchange="GLOBEX", currency="USD")
            bars = ib.reqHistoricalData(
                contract, endDateTime="", durationStr="1 Y",
                barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=False, formatDate=1,
            )
            df = util.df(bars)
            if not df.empty:
                df["contract"] = expiry
                all_dfs.append(df)
                print(f"  ✓ {expiry}: {len(df)} rows")
            else:
                print(f"  ⚠ {expiry}: no data")
            time.sleep(2.5)
        except Exception as e:
            print(f"  ✗ {expiry}: {e}")

    ib.disconnect()

    if all_dfs:
        combined = pd.concat(all_dfs).set_index("date").sort_index()
        if save:
            combined.to_csv("data/6E_continuous.csv")
            print(f"  Saved data/6E_continuous.csv ({len(combined)} rows)")
        return combined
    return pd.DataFrame()


# ════════════════════════════════════════════════════════════════
# SECTION 3 — TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Rolling Bollinger Bands.

    Returns
    -------
    (mean, upper, lower) as pd.Series
    """
    mean  = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = mean + num_std * std
    lower = mean - num_std * std
    return mean, upper, lower


def regression_slope(series: pd.Series, window: int = 5) -> pd.Series:
    """
    Rolling linear regression slope (vectorized).

    Returns slope in same units as series per bar.

    Note: slope is almost always negative when price is below the lower BB
    (price had to fall to get there), so using slope>0 as a LONG filter
    generates zero signals. Use regime_mode="trend_aware" instead.
    """
    arr = series.values.astype(float)
    n   = len(arr)
    slopes = np.full(n, np.nan)
    x      = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var  = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        if not np.any(np.isnan(y)):
            slopes[i] = ((x - x_mean) * (y - y.mean())).sum() / x_var
    return pd.Series(slopes, index=series.index)


def realized_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Rolling historical (realized) volatility.

    Parameters
    ----------
    returns   : Daily return series (pct_change)
    window    : Look-back in trading days (21 ≈ 1 month, 63 ≈ 1 quarter)
    annualize : Multiply by sqrt(252) if True

    Returns
    -------
    Annualized rolling vol
    """
    rv = returns.rolling(window).std()
    return rv * np.sqrt(252) if annualize else rv


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (volatility range measure)."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def zscore(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def rolling_correlation(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return s1.rolling(window).corr(s2)


# ════════════════════════════════════════════════════════════════
# SECTION 4 — MEAN-REVERSION STRATEGY
# ════════════════════════════════════════════════════════════════

def apply_mean_reversion_strategy(
    df: pd.DataFrame,
    window: int          = 20,
    slope_window: int    = 5,
    num_std: float       = 2.0,
    ma_trend_window: int = 200,
    slope_threshold: float = 0.0,
    require_regime_filter: bool = True,
    regime_mode: str     = "none",
) -> pd.DataFrame:
    """
    Bollinger Band mean-reversion strategy with optional regime filter.

    Regime Modes
    ------------
    "none"         : Fire on any BB pierce regardless of macro trend.
                     → Pure mean-reversion. Best for EUR/USD (Sharpe 0.60).

    "trend_aware"  : Long only when price > 200d MA (buy dips in uptrends);
                     Short only when price < 200d MA (sell rallies in downtrends).
                     → Lower signal count, lower drawdown. Best for GBP, AUD.

    "soft"         : Allows entries within 5% of the 200d MA in either direction.
                     → Intermediate signal count.

    Signal Logic (applies to all modes)
    ------------------------------------
    Long  (signal= 1): close < lower_band  [+ regime condition]
    Short (signal=-1): close > upper_band  [+ regime condition]

    Key Fix vs v1
    -------------
    v1 used slope>0 as a long filter. The slope of the last N bars is ALWAYS
    negative when price is below the lower band (price fell to get there).
    This generated ZERO signals on 20 years of real data. The regime_mode
    approach replaces slope-based filtering correctly.
    """
    df = df.copy()

    # Bollinger Bands
    df["bb_mean"], df["upper"], df["lower"] = bollinger_bands(
        df["close"], window=window, num_std=num_std
    )

    # Slope (kept for analytics/reference; not used for signal filtering)
    df["slope"] = regression_slope(df["close"], slope_window) / df["close"]

    # 200d MA trend indicator
    df["ma_trend"] = df["close"].rolling(ma_trend_window).mean()

    # %B and band width (diagnostics)
    band_range       = (df["upper"] - df["lower"]).replace(0, np.nan)
    df["pct_b"]      = (df["close"] - df["lower"]) / band_range
    df["band_width"] = band_range / df["bb_mean"]

    # ── Signal generation ──
    df["signal"] = 0

    if regime_mode == "trend_aware":
        long_cond  = (df["close"] < df["lower"]) & (df["close"] > df["ma_trend"])
        short_cond = (df["close"] > df["upper"]) & (df["close"] < df["ma_trend"])
    elif regime_mode == "soft":
        long_cond  = (df["close"] < df["lower"]) & (df["close"] > df["ma_trend"] * 0.95)
        short_cond = (df["close"] > df["upper"]) & (df["close"] < df["ma_trend"] * 1.05)
    else:  # "none" — pure BB
        long_cond  = df["close"] < df["lower"]
        short_cond = df["close"] > df["upper"]

    df.loc[long_cond,  "signal"] = 1
    df.loc[short_cond, "signal"] = -1

    # Signal strength (0–1): how far outside the band
    df["signal_strength"] = 0.0
    df.loc[long_cond,  "signal_strength"] = (
        (df.loc[long_cond,  "lower"] - df.loc[long_cond,  "close"]) / df.loc[long_cond,  "lower"]
    ).clip(0, 0.05) / 0.05
    df.loc[short_cond, "signal_strength"] = (
        (df.loc[short_cond, "close"] - df.loc[short_cond, "upper"]) / df.loc[short_cond, "upper"]
    ).clip(0, 0.05) / 0.05

    return df


# ════════════════════════════════════════════════════════════════
# SECTION 5 — VRP SIGNAL
# ════════════════════════════════════════════════════════════════

def compute_realized_vol(df: pd.DataFrame, rv_window: int = 21) -> pd.DataFrame:
    """Add rolling realized vol columns to df."""
    df = df.copy()
    ret = df["close"].pct_change()
    df["returns"] = ret
    df["rv_5"]    = realized_volatility(ret, window=5)
    df["rv_21"]   = realized_volatility(ret, window=rv_window)
    df["rv_63"]   = realized_volatility(ret, window=63)
    return df


def estimate_iv_proxy(df: pd.DataFrame, asset: str = "EURUSD",
                      vrp_structural_premium: float = 0.02) -> pd.DataFrame:
    """
    Estimate implied volatility via structural prior + live yfinance options chain.

    Structural prior: IV = RV * (1 + premium) + premium
    This embeds the empirical finding that FX IV > RV on average by ~1–3%.

    For recent data (~30d), attempts to pull actual ATM IV from the
    corresponding FX ETF options chain (FXE, FXY, FXB, FXA).
    """
    df = df.copy()
    df["iv_proxy"] = df["rv_21"] * (1 + vrp_structural_premium) + vrp_structural_premium

    if YFINANCE_AVAILABLE:
        etf = ASSETS.get(asset, {}).get("etf", "FXE")
        try:
            ticker    = yf.Ticker(etf)
            exp_dates = ticker.options
            if exp_dates:
                target      = pd.Timestamp.now() + pd.Timedelta(days=30)
                nearest_exp = min(exp_dates, key=lambda d: abs((pd.Timestamp(d) - target).days))
                chain       = ticker.option_chain(nearest_exp)
                atm_calls   = chain.calls[
                    (chain.calls["impliedVolatility"] > 0.001) &
                    (~chain.calls["inTheMoney"])
                ]
                if not atm_calls.empty:
                    recent_iv  = atm_calls["impliedVolatility"].median()
                    cutoff     = df.index[-1] - pd.Timedelta(days=45)
                    df.loc[df.index >= cutoff, "iv_proxy"] = recent_iv
        except Exception:
            pass

    return df


def compute_vrp(df: pd.DataFrame, asset: str = "EURUSD", rv_window: int = 21) -> pd.DataFrame:
    """
    Full VRP pipeline.

    Adds columns:
    ─────────────
    rv_5, rv_21, rv_63  : Realized vols at different horizons
    iv_proxy            : Estimated implied volatility
    vrp                 : IV - RV spread (the premium sellers earn)
    vrp_zscore          : 63-day rolling z-score of VRP
    vrp_regime          :  1 = sell vol (IV overpriced)
                          -1 = buy vol  (IV cheap)
                           0 = neutral
    vrp_size_mult       : 0.5–1.5 position multiplier based on VRP conviction
    """
    df = compute_realized_vol(df, rv_window=rv_window)
    df = estimate_iv_proxy(df, asset=asset,
                           vrp_structural_premium=ASSETS.get(asset, {}).get("vrp_premium", 0.02))
    df["vrp"]        = df["iv_proxy"] - df["rv_21"]
    df["vrp_zscore"] = zscore(df["vrp"], window=63)
    df["vrp_regime"] = 0
    df.loc[df["vrp"] >  0.005, "vrp_regime"] =  1
    df.loc[df["vrp"] < -0.005, "vrp_regime"] = -1
    vrp_norm             = df["vrp_zscore"].clip(-2, 2) / 2
    df["vrp_size_mult"]  = 1.0 + 0.5 * vrp_norm.fillna(0)
    return df


def apply_vrp_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach VRP-aware position sizing to existing signal column.

    Adds:
    ─────
    vrp_filtered_signal : same as signal (direction unchanged)
    position_size       : signal_strength × vrp_size_mult (0–2)
    """
    df = df.copy()
    df["vrp_filtered_signal"] = df["signal"]
    base_size         = df["signal"].abs() * (0.5 + 0.5 * df.get(
        "signal_strength", pd.Series(1.0, index=df.index)
    ))
    df["position_size"] = (base_size * df["vrp_size_mult"]).clip(0, 2)
    return df


# ════════════════════════════════════════════════════════════════
# SECTION 6 — BLACK-SCHOLES MODEL & GREEKS
# ════════════════════════════════════════════════════════════════

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """
    Black-Scholes option price.

    Parameters
    ----------
    S     : Spot price
    K     : Strike price
    T     : Time to maturity in years
    r     : Risk-free rate (annualized)
    sigma : Volatility (annualized)
    """
    if T <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> dict:
    """
    All Black-Scholes Greeks.

    Returns dict with: delta, gamma, theta (per day), vega (per 1% vol), rho
    """
    if T <= 0:
        return dict(delta=0, gamma=0, theta=0, vega=0, rho=0)
    d1  = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega  = S * nd1 * np.sqrt(T) / 100
    if option_type == "call":
        delta = Nd1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * Nd2) / 365
        rho   = K * T * np.exp(-r * T) * Nd2 / 100
    else:
        delta = Nd1 - 1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return dict(delta=round(delta, 4), gamma=round(gamma, 6),
                theta=round(theta, 6), vega=round(vega, 6), rho=round(rho, 6))


def bs_iv(market_price: float, S: float, K: float, T: float, r: float,
          option_type: str = "call") -> float:
    """
    Implied volatility from market price (Brent's method).

    Returns np.nan if no solution found.
    """
    if T <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price < intrinsic:
        return np.nan
    try:
        return round(brentq(lambda s: bs_price(S, K, T, r, s, option_type) - market_price,
                            1e-6, 10.0, xtol=1e-6, maxiter=200), 6)
    except (ValueError, RuntimeError):
        return np.nan


# ════════════════════════════════════════════════════════════════
# SECTION 7 — OPTIONS BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def simulate_spread(S: float, K1: float, K2: float, T: float, r: float, sigma: float,
                    direction: str = "bull_call") -> dict:
    """Entry-cost-only spread simulation (v1 compatible)."""
    if direction == "bull_call":
        net = bs_price(S, K1, T, r, sigma, "call") - bs_price(S, K2, T, r, sigma, "call")
    elif direction == "bear_put":
        net = bs_price(S, K2, T, r, sigma, "put") - bs_price(S, K1, T, r, sigma, "put")
    else:
        raise ValueError("direction must be 'bull_call' or 'bear_put'")
    return {"net_premium": net}


def simulate_spread_pnl(S_entry: float, S_expiry: float, K1: float, K2: float,
                        T_entry: float, r: float, sigma_entry: float,
                        direction: str = "bull_call") -> dict:
    """
    Full spread P&L from entry to expiry (v2 — fixes v1 which only computed entry cost).

    Entry cost  = net premium paid at trade open
    Expiry value = intrinsic value at expiry (capped spread payoff)
    PnL          = expiry_value − entry_cost

    Parameters
    ----------
    S_entry   : Spot at entry
    S_expiry  : Spot at expiry
    K1, K2    : Lower and upper strike
    T_entry   : Time to expiry at entry (years)
    r         : Risk-free rate
    sigma_entry : Volatility at entry (use rolling RV)
    direction : 'bull_call' | 'bear_put'
    """
    if direction == "bull_call":
        entry_cost   = (bs_price(S_entry, K1, T_entry, r, sigma_entry, "call") -
                        bs_price(S_entry, K2, T_entry, r, sigma_entry, "call"))
        expiry_value = max(0.0, S_expiry - K1) - max(0.0, S_expiry - K2)
    elif direction == "bear_put":
        entry_cost   = (bs_price(S_entry, K2, T_entry, r, sigma_entry, "put") -
                        bs_price(S_entry, K1, T_entry, r, sigma_entry, "put"))
        expiry_value = max(0.0, K2 - S_expiry) - max(0.0, K1 - S_expiry)
    else:
        raise ValueError("direction must be 'bull_call' or 'bear_put'")

    pnl    = expiry_value - entry_cost
    pnl_pct = pnl / entry_cost if entry_cost != 0 else 0.0
    return dict(
        entry_cost=round(entry_cost, 6), expiry_value=round(expiry_value, 6),
        pnl=round(pnl, 6), pnl_pct=round(pnl_pct, 4),
        max_profit=round(K2 - K1 - entry_cost, 6), max_loss=round(entry_cost, 6),
        direction=direction, K1=K1, K2=K2,
    )


def run_options_strategy(df: pd.DataFrame, r: float = 0.025, T: float = 1/12,
                         otm_pct: float = 0.01, use_dynamic_sigma: bool = True,
                         apply_vrp_filter_flag: bool = True, holding_bars: int = 21):
    """
    Run options spread strategy with full expiry P&L simulation.

    Key fixes vs v1:
    ─────────────────
    • Dynamic sigma: uses rv_21 column (was hardcoded 0.12)
    • Expiry simulation: computes intrinsic value at expiry (v1 only had entry cost)
    • VRP filter: skips trades when VRP ≤ 0 (options are cheap — don't sell)
    • Holding period: simulates 21 trading days (1 month) to expiry

    Parameters
    ----------
    df              : DataFrame with 'close', 'signal', optionally 'rv_21', 'vrp'
    r               : Risk-free rate
    T               : Time to expiry at entry (years)
    otm_pct         : OTM distance for spread legs (1% = 0.01)
    use_dynamic_sigma: Use rv_21 column as sigma if available
    apply_vrp_filter_flag: Only enter when VRP > 0
    holding_bars    : Trading days to simulate holding to expiry

    Returns
    -------
    (trade_log DataFrame, equity Series)
    """
    results = []
    dates   = df.index

    for i, (date, row) in enumerate(df.iterrows()):
        if row["signal"] == 0:
            continue

        spot = row["close"]

        # Dynamic sigma
        sigma = 0.08
        if use_dynamic_sigma and "rv_21" in df.columns:
            rv = row["rv_21"]
            if not (pd.isna(rv) or rv <= 0.005):
                sigma = rv

        # VRP filter
        if apply_vrp_filter_flag and "vrp" in df.columns:
            vrp = row.get("vrp", 0)
            if pd.isna(vrp) or vrp <= 0:
                continue

        # Strike selection
        if row["signal"] == 1:   # Expect rise → Bull Call
            K1, K2    = spot * (1 - otm_pct), spot * (1 + otm_pct)
            direction = "bull_call"
        else:                     # Expect fall → Bear Put
            K1, K2    = spot * (1 - otm_pct), spot
            direction = "bear_put"

        # Expiry spot price
        exp_idx  = min(i + holding_bars, len(df) - 1)
        S_expiry = df["close"].iloc[exp_idx]

        trade = simulate_spread_pnl(spot, S_expiry, K1, K2, T, r, sigma, direction)
        results.append(dict(
            entry_date=date, expiry_date=dates[exp_idx],
            spot_entry=round(spot, 5), spot_expiry=round(S_expiry, 5),
            signal=row["signal"], direction=direction,
            K1=round(K1, 5), K2=round(K2, 5), sigma_used=round(sigma, 4),
            **{k: trade[k] for k in ["entry_cost", "expiry_value", "pnl", "pnl_pct"]},
            win=trade["pnl"] > 0,
        ))

    if not results:
        return pd.DataFrame(), pd.Series(dtype=float)

    res_df = pd.DataFrame(results).set_index("entry_date")
    res_df["cumulative_pnl"] = res_df["pnl"].cumsum()
    mean_cost = res_df["entry_cost"].mean()
    res_df["equity"] = 100 + res_df["cumulative_pnl"] / (mean_cost if mean_cost else 1) * 100
    return res_df, res_df["equity"]


# ════════════════════════════════════════════════════════════════
# SECTION 8 — DIRECTIONAL BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def simulate_trades(df: pd.DataFrame, transaction_cost_pct: float = 0.0001,
                    use_position_sizing: bool = False,
                    initial_capital: float = 100.0) -> pd.DataFrame:
    """
    Vectorized trade simulation with transaction costs.

    Parameters
    ----------
    df                    : DataFrame with 'close' and 'signal' columns
    transaction_cost_pct  : Cost per position change (1 pip ≈ 0.0001)
    use_position_sizing   : Use 'position_size' column for fractional sizing
    initial_capital       : Starting equity value

    Adds columns
    ────────────
    position        : Lagged signal (enter next bar)
    returns         : Daily price returns
    strategy_returns: Gross position × returns
    costs           : Transaction costs on position changes
    net_returns     : strategy_returns − costs
    equity          : Cumulative equity curve
    """
    df = df.copy()

    if use_position_sizing and "position_size" in df.columns:
        df["position"] = df["position_size"].shift(1).fillna(0)
    else:
        df["position"] = df["signal"].shift(1).fillna(0)

    df["returns"]          = df["close"].pct_change()
    df["strategy_returns"] = df["returns"] * df["position"]
    df["position_change"]  = df["position"].diff().abs()
    df["costs"]            = df["position_change"] * transaction_cost_pct
    df["net_returns"]      = df["strategy_returns"] - df["costs"]
    df["equity"]           = (1 + df["net_returns"]).cumprod() * initial_capital
    return df


def compute_metrics(df: pd.DataFrame, returns_col: str = "net_returns",
                    equity_col: str = "equity", label: str = "Strategy") -> dict:
    """
    Comprehensive performance metrics.

    Returns dict with: label, total_return, cagr, sharpe, sortino, calmar,
    max_drawdown, ann_volatility, win_rate, n_trades, best_day, worst_day,
    skewness, kurtosis, years, final_equity
    """
    ret    = df[returns_col].dropna()
    equity = df[equity_col].dropna()
    if len(ret) < 20:
        return {"label": label, "error": "insufficient data"}

    initial     = equity.iloc[0]
    final       = equity.iloc[-1]
    total_ret   = final / initial - 1
    n_years     = (df.index[-1] - df.index[0]).days / 365.25
    cagr        = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe      = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    sortino     = ret.mean() / ret[ret < 0].std() * np.sqrt(252) if ret[ret < 0].std() > 0 else 0
    roll_max    = equity.cummax()
    max_dd      = ((equity - roll_max) / roll_max).min()
    calmar      = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate    = (ret > 0).sum() / (ret != 0).sum() if (ret != 0).sum() > 0 else 0
    n_trades    = int((df.get("position", df["signal"]).diff().abs() > 0).sum()) if "position" in df else 0
    return dict(
        label=label, total_return=round(total_ret * 100, 2), cagr=round(cagr * 100, 2),
        sharpe=round(sharpe, 3), sortino=round(sortino, 3), calmar=round(calmar, 3),
        max_drawdown=round(max_dd * 100, 2), ann_volatility=round(ret.std() * np.sqrt(252) * 100, 2),
        win_rate=round(win_rate * 100, 1), n_trades=n_trades,
        best_day=round(ret.max() * 100, 2), worst_day=round(ret.min() * 100, 2),
        skewness=round(ret.skew(), 3), kurtosis=round(ret.kurt(), 3),
        years=round(n_years, 1), final_equity=round(final, 2),
    )


def print_metrics(m: dict) -> None:
    """Pretty-print metrics dict."""
    lbl = m.get("label", "Strategy")
    print(f"\n{'═'*52}")
    print(f"  {lbl}")
    print(f"{'═'*52}")
    if "error" in m:
        print(f"  ERROR: {m['error']}")
        return
    print(f"  Total Return   : {m['total_return']:>8.2f}%")
    print(f"  CAGR           : {m['cagr']:>8.2f}%")
    print(f"  Sharpe         : {m['sharpe']:>8.3f}")
    print(f"  Sortino        : {m['sortino']:>8.3f}")
    print(f"  Calmar         : {m['calmar']:>8.3f}")
    print(f"  Max Drawdown   : {m['max_drawdown']:>8.2f}%")
    print(f"  Ann Volatility : {m['ann_volatility']:>8.2f}%")
    print(f"  Win Rate       : {m['win_rate']:>8.1f}%")
    print(f"  No. of Trades  : {m['n_trades']:>8d}")
    print(f"  Skewness       : {m['skewness']:>8.3f}")
    print(f"  Kurtosis       : {m['kurtosis']:>8.3f}")
    print(f"  Period (years) : {m['years']:>8.1f}")
    print(f"  Final Equity   : ${m['final_equity']:>8.2f}")
    print(f"{'═'*52}\n")


def walk_forward_split(df: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7):
    """
    Walk-forward train/test generator.

    Yields
    ------
    (train_df, test_df, label_str) tuples
    """
    n          = len(df)
    split_size = n // n_splits
    for i in range(n_splits):
        start  = i * split_size
        end    = start + split_size if i < n_splits - 1 else n
        chunk  = df.iloc[start:end]
        n_tr   = int(len(chunk) * train_ratio)
        train, test = chunk.iloc[:n_tr], chunk.iloc[n_tr:]
        label = f"Split {i+1}: {train.index[0].date()} → {test.index[-1].date()}"
        if len(test) > 20:
            yield train, test, label


# ════════════════════════════════════════════════════════════════
# SECTION 9 — MULTI-ASSET PORTFOLIO
# ════════════════════════════════════════════════════════════════

def inverse_vol_weights(returns_dict: dict, vol_window: int = 63) -> pd.DataFrame:
    """
    Inverse-volatility weights (equal risk contribution).

    Parameters
    ----------
    returns_dict : {asset_name: returns_series}
    vol_window   : Rolling window for vol estimation

    Returns
    -------
    DataFrame of weights (rows=dates, columns=asset names)
    """
    ret_df   = pd.DataFrame(returns_dict).dropna(how="all")
    vols     = ret_df.rolling(vol_window).std() * np.sqrt(252)
    inv_vols = 1.0 / vols.replace(0, np.nan)
    return inv_vols.div(inv_vols.sum(axis=1), axis=0)


def run_portfolio(asset_dfs: dict, transaction_cost_pct: float = 0.0001,
                  initial_capital: float = 100.0, weighting: str = "inv_vol",
                  vol_window: int = 63):
    """
    Multi-asset portfolio with configurable weighting scheme.

    Parameters
    ----------
    asset_dfs   : {name: df_with_net_returns_column}
    weighting   : "inv_vol" | "equal"
    vol_window  : Used only for "inv_vol"

    Returns
    -------
    (portfolio_df, weights_df)
    """
    ret_dict = {n: d.get("net_returns", d.get("strategy_returns"))
                for n, d in asset_dfs.items()
                if "net_returns" in d or "strategy_returns" in d}

    ret_df = pd.DataFrame(ret_dict).fillna(0)

    if weighting == "inv_vol":
        w = inverse_vol_weights(ret_dict, vol_window=vol_window)
        w = w.shift(1).reindex(ret_df.index).fillna(1.0 / len(ret_dict))
    else:
        w = pd.DataFrame(1.0 / len(ret_dict), index=ret_df.index, columns=ret_df.columns)

    port_ret   = (ret_df * w).sum(axis=1)
    port_costs = w.diff().abs().sum(axis=1) * transaction_cost_pct
    port_net   = port_ret - port_costs

    port_df = pd.DataFrame({
        "portfolio_returns": port_ret,
        "portfolio_costs":   port_costs,
        "portfolio_net":     port_net,
        "equity":            (1 + port_net).cumprod() * initial_capital,
    })
    for n, r in ret_dict.items():
        port_df[f"ret_{n}"] = r.reindex(port_df.index).fillna(0)

    return port_df, w


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    rm = returns.rolling(window).mean()
    rs = returns.rolling(window).std()
    return (rm / rs * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)


# ════════════════════════════════════════════════════════════════
# SECTION 10 — PARAMETER OPTIMIZATION (WALK-FORWARD)
# ════════════════════════════════════════════════════════════════

def _quick_backtest(df: pd.DataFrame, window: int, num_std: float, regime: str,
                    tc: float = 0.0001) -> dict | None:
    """Run one parameter combo — used internally by optimizer."""
    df = df.copy()
    bb_mean, upper, lower = bollinger_bands(df["close"], window=window, num_std=num_std)
    ma200 = df["close"].rolling(200).mean()

    if regime == "trend_aware":
        long_c  = (df["close"] < lower) & (df["close"] > ma200)
        short_c = (df["close"] > upper) & (df["close"] < ma200)
    else:
        long_c  = df["close"] < lower
        short_c = df["close"] > upper

    df["signal"]   = 0
    df.loc[long_c,  "signal"] = 1
    df.loc[short_c, "signal"] = -1
    df["position"] = df["signal"].shift(1).fillna(0)
    df["ret"]      = df["close"].pct_change()
    df["strat"]    = df["ret"] * df["position"]
    df["costs"]    = df["position"].diff().abs() * tc
    df["net"]      = df["strat"] - df["costs"]
    df["equity"]   = (1 + df["net"]).cumprod() * 100

    net = df["net"].dropna()
    if len(net) < 100 or net.std() == 0:
        return None
    sharpe = net.mean() / net.std() * np.sqrt(252)
    n_yrs  = (df.index[-1] - df.index[0]).days / 365.25
    cagr   = (df["equity"].iloc[-1] / 100) ** (1 / n_yrs) - 1
    max_dd = ((df["equity"].cummax() - df["equity"]) / df["equity"].cummax()).max()
    return dict(sharpe=sharpe, cagr=cagr * 100, max_dd=max_dd * 100,
                n_trades=int((df["position"].diff().abs() > 0).sum()))


def run_walk_forward_optimization(df: pd.DataFrame, asset: str = "EURUSD",
                                  n_splits: int = 5) -> pd.DataFrame:
    """
    Walk-forward parameter search.

    For each fold: optimize on train set, evaluate on test set (OOS only).
    Reports OOS Sharpe, CAGR, MaxDD, and the best parameter set per fold.

    Parameters
    ----------
    df       : Full price DataFrame
    asset    : Asset name (used in labels)
    n_splits : Number of walk-forward folds

    Returns
    -------
    DataFrame of OOS results per fold
    """
    combos  = list(itertools.product(
        PARAM_GRID["window"], PARAM_GRID["num_std"], PARAM_GRID["regime"]
    ))
    n       = len(df)
    fold_sz = n // n_splits
    records = []

    for fold in range(n_splits):
        tr_end  = fold_sz * (fold + 1)
        ts_end  = min(tr_end + fold_sz // 2, n)
        if tr_end >= n - 50:
            break
        train, test = df.iloc[:tr_end], df.iloc[tr_end:ts_end]
        if len(test) < 50:
            continue

        best_s, best_p = -999, None
        for w, s, r in combos:
            res = _quick_backtest(train, w, s, r)
            if res and res["sharpe"] > best_s:
                best_s, best_p = res["sharpe"], (w, s, r)

        if not best_p:
            continue
        oos = _quick_backtest(test, *best_p)
        if oos:
            oos.update(fold=fold + 1, window=best_p[0], num_std=best_p[1],
                       regime=best_p[2], train_sharpe=round(best_s, 3))
            records.append(oos)

    return pd.DataFrame(records)


def full_grid_search(df: pd.DataFrame, tc: float = 0.0001) -> pd.DataFrame:
    """
    In-sample grid search across all parameter combos.
    For reference only — never use in-sample results for trading decisions.
    """
    records = []
    for w, s, r in itertools.product(
        PARAM_GRID["window"], PARAM_GRID["num_std"], PARAM_GRID["regime"]
    ):
        res = _quick_backtest(df, w, s, r, tc=tc)
        if res:
            res.update(window=w, num_std=s, regime=r)
            records.append(res)
    return pd.DataFrame(records).sort_values("sharpe", ascending=False)


# ════════════════════════════════════════════════════════════════
# SECTION 11 — MONTE CARLO
# ════════════════════════════════════════════════════════════════

def monte_carlo(returns: pd.Series, n_simulations: int = 5000,
                label: str = "Strategy") -> dict:
    """
    Bootstrap Monte Carlo simulation.

    Resamples daily returns with replacement N times to estimate the
    distribution of Sharpe, CAGR, and Max Drawdown.

    The P5 Sharpe (5th percentile worst case) is the key robustness metric.
    A strategy is considered robust if P5 Sharpe > 0.

    Parameters
    ----------
    returns       : Daily net returns series
    n_simulations : Number of bootstrap paths

    Returns
    -------
    dict with 'sharpe_pct', 'cagr_pct', 'maxdd_pct' each containing
    {5: val, 25: val, 50: val, 75: val, 95: val}
    """
    arr = returns.dropna().values
    if len(arr) < 50:
        return {}

    sharpes, cagrs, mds = [], [], []
    for _ in range(n_simulations):
        s   = np.random.choice(arr, size=len(arr), replace=True)
        eq  = (1 + s).cumprod()
        sharpes.append(s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0)
        cagrs.append((eq[-1] ** (252 / len(s)) - 1) * 100)
        mds.append(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max() * 100)

    pcts = [5, 25, 50, 75, 95]
    return dict(
        label=label,
        sharpe_pct={p: round(np.percentile(sharpes, p), 3) for p in pcts},
        cagr_pct=  {p: round(np.percentile(cagrs,   p), 2) for p in pcts},
        maxdd_pct= {p: round(np.percentile(mds,      p), 2) for p in pcts},
    )


def print_monte_carlo(mc: dict) -> None:
    """Pretty-print Monte Carlo results."""
    if not mc:
        return
    print(f"\n  Monte Carlo — {mc.get('label', '')} (simulations)")
    print(f"  {'Pct':>5}  {'Sharpe':>8}  {'CAGR%':>8}  {'MaxDD%':>8}")
    print(f"  {'─'*38}")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:>2d}   {mc['sharpe_pct'][p]:>8.3f}  "
              f"{mc['cagr_pct'][p]:>8.2f}  {mc['maxdd_pct'][p]:>8.2f}")


# ════════════════════════════════════════════════════════════════
# SECTION 12 — PLOTTING
# ════════════════════════════════════════════════════════════════

def plot_asset(name: str, df: pd.DataFrame, save_dir: str = "results/plots") -> str:
    """4-panel chart: price/bands/signals, equity/DD, VRP, rolling Sharpe."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), gridspec_kw={"hspace": 0.4})
    c = COLORS.get(name, "#1f77b4")

    # 1. Price + bands + signals
    ax = axes[0]
    ax.plot(df["close"], color=c, lw=0.6, label=name)
    ax.plot(df["upper"], "--", color="gray", lw=0.4, alpha=0.7)
    ax.plot(df["lower"], "--", color="gray", lw=0.4, alpha=0.7)
    ax.plot(df.get("ma_trend", pd.Series()), color="orange", lw=0.7, alpha=0.6, label="200d MA")
    longs  = df[df["signal"] ==  1]
    shorts = df[df["signal"] == -1]
    ax.scatter(longs.index,  longs["close"],  marker="^", color="green", s=12, zorder=5,
               label=f"Long ({len(longs)})")
    ax.scatter(shorts.index, shorts["close"], marker="v", color="red",   s=12, zorder=5,
               label=f"Short ({len(shorts)})")
    ax.set_title(f"{name} — Price, Bollinger Bands & Signals", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=4)

    # 2. Equity + drawdown
    ax2 = axes[1]
    ax2.plot(df["equity"], color="#2ca02c", lw=1.2, label="Equity")
    ax2.axhline(100, color="gray", lw=0.5, ls="--")
    roll_max = df["equity"].cummax()
    dd       = (df["equity"] - roll_max) / roll_max * 100
    ax2r     = ax2.twinx()
    ax2r.fill_between(df.index, dd, 0, color="red", alpha=0.2)
    ax2r.set_ylabel("DD%", color="red", fontsize=8)
    ax2r.tick_params(axis="y", labelcolor="red")
    ax2.set_title(f"{name} — Equity Curve & Drawdown", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    # 3. VRP
    ax3 = axes[2]
    if "vrp" in df.columns:
        v = df["vrp"].dropna() * 100
        ax3.plot(v, color="#9467bd", lw=0.5)
        ax3.axhline(0, color="black", lw=0.5)
        ax3.fill_between(v.index, v, 0, where=v > 0, alpha=0.3, color="green",  label="Sell Vol")
        ax3.fill_between(v.index, v, 0, where=v < 0, alpha=0.3, color="red",    label="Buy Vol")
        ax3.legend(fontsize=8)
    ax3.set_title(f"{name} — VRP = IV − RV (%)", fontsize=11, fontweight="bold")

    # 4. Rolling Sharpe
    ax4 = axes[3]
    rs = rolling_sharpe(df.get("net_returns", df.get("strategy_returns", pd.Series())).fillna(0))
    ax4.plot(rs, color="#d62728", lw=0.8)
    ax4.axhline(0,   color="black", lw=0.5)
    ax4.axhline(0.5, color="green", lw=0.5, ls="--", label="0.5 target")
    ax4.fill_between(rs.index, rs, 0, where=rs > 0, alpha=0.2, color="green")
    ax4.fill_between(rs.index, rs, 0, where=rs < 0, alpha=0.2, color="red")
    ax4.set_title(f"{name} — Rolling 1Y Sharpe", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8)

    path = os.path.join(save_dir, f"{name}_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


def plot_portfolio(portfolio_df: pd.DataFrame, weights_df: pd.DataFrame,
                   asset_results: dict, save_dir: str = "results/plots") -> str:
    """3-panel portfolio chart: equity comparison, drawdown, asset weights."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"hspace": 0.4})

    ax = axes[0]
    ax.plot(portfolio_df["equity"], color="black", lw=2, label="Portfolio", zorder=5)
    for name, df in asset_results.items():
        eq = df["equity"].reindex(portfolio_df.index, method="ffill")
        ax.plot(eq, lw=0.7, alpha=0.55, color=COLORS.get(name, None), label=name)
    ax.axhline(100, color="gray", lw=0.5, ls="--")
    ax.set_title("Multi-Asset Portfolio vs Individual Assets", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    rm = portfolio_df["equity"].cummax()
    dd = (portfolio_df["equity"] - rm) / rm * 100
    ax2.fill_between(portfolio_df.index, dd, 0, color="red", alpha=0.5)
    ax2.set_title("Portfolio Drawdown (%)", fontsize=12, fontweight="bold")

    ax3 = axes[2]
    weights_df.reindex(portfolio_df.index).fillna(0).plot.area(ax=ax3, alpha=0.7, colormap="tab10")
    ax3.set_title("Inverse-Vol Asset Weights Over Time", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1)

    path = os.path.join(save_dir, "portfolio_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


def plot_monte_carlo(mc: dict, save_dir: str = "results/plots") -> str:
    """3-panel histogram: Sharpe, CAGR, MaxDD distributions."""
    os.makedirs(save_dir, exist_ok=True)
    label = mc.get("label", "Strategy")

    # Need raw simulation data — re-run quickly
    # This version plots percentile lines on placeholder distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Monte Carlo — {label}", fontsize=12, fontweight="bold")

    for ax, key, col, xlabel in zip(
        axes,
        ["sharpe_pct", "cagr_pct", "maxdd_pct"],
        ["#1f77b4", "#2ca02c", "#d62728"],
        ["Sharpe", "CAGR (%)", "Max Drawdown (%)"]
    ):
        pcts = mc[key]
        vals = list(pcts.values())
        ax.bar([f"P{p}" for p in pcts.keys()], vals, color=col, alpha=0.75, edgecolor="white")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(xlabel, fontsize=10)
        ax.set_xlabel("Percentile")
        for i, (p, v) in enumerate(pcts.items()):
            ax.text(i, v + abs(v) * 0.02, f"{v:.2f}", ha="center", fontsize=8)

    path = os.path.join(save_dir, "monte_carlo.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


# ════════════════════════════════════════════════════════════════
# SECTION 13 — FULL PIPELINE (MASTER RUNNER)
# ════════════════════════════════════════════════════════════════

def run_full_pipeline(verbose: bool = True, save_plots: bool = True,
                      save_results: bool = True) -> dict:
    """
    Run the complete VRP strategy pipeline.

    Steps
    ─────
    1. Load data for all configured assets
    2. Apply mean-reversion strategy (per-asset optimal params)
    3. Compute VRP signal
    4. Directional backtest with transaction costs
    5. Options strategy (EURUSD)
    6. Multi-asset portfolio (inv-vol weighting)
    7. Walk-forward validation (EURUSD)
    8. Monte Carlo simulation (EURUSD)
    9. Save results and plots

    Returns
    -------
    dict with keys: asset_results, asset_metrics, portfolio_df, options_results,
                    wf_results, mc_results
    """
    print("\n" + "═" * 62)
    print("  VRP STRATEGY — FULL PIPELINE (v2.0)")
    print("═" * 62)

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # ── 1. Load data ──
    print("\n[1] Loading data...")
    raw_dfs = load_all_assets()

    # ── 2–4. Strategy + VRP + backtest ──
    print("\n[2] Running per-asset strategies...")
    asset_results = {}
    asset_metrics = {}

    for name, df in raw_dfs.items():
        p = OPTIMAL_PARAMS.get(name, {"window": 20, "num_std": 2.0, "regime_mode": "none"})
        df = apply_mean_reversion_strategy(df, window=p["window"], num_std=p["num_std"],
                                           regime_mode=p["regime_mode"])
        df["returns"] = df["close"].pct_change()
        df = compute_vrp(df, asset=name)
        df = apply_vrp_filter(df)
        df = simulate_trades(df, transaction_cost_pct=BACKTEST_CFG["transaction_cost_pct"])
        asset_results[name] = df
        asset_metrics[name] = compute_metrics(df, label=name)
        if verbose:
            print_metrics(asset_metrics[name])
        if save_plots:
            plot_asset(name, df)

    # ── 5. Options (EURUSD) ──
    print("\n[3] Running options strategy (EURUSD)...")
    opt_results = pd.DataFrame()
    if "EURUSD" in asset_results:
        opt_results, _ = run_options_strategy(asset_results["EURUSD"], **OPTIONS_CFG)
        if not opt_results.empty:
            wr = opt_results["win"].mean() * 100
            sharpe_opt = (opt_results["pnl"].mean() / opt_results["pnl"].std() * np.sqrt(12)
                         if opt_results["pnl"].std() > 0 else 0)
            print(f"  Trades: {len(opt_results)} | Win Rate: {wr:.1f}% | Sharpe: {sharpe_opt:.3f}")
            if save_results:
                opt_results.to_csv("results/options_trades.csv")

    # ── 6. Portfolio ──
    print("\n[4] Building multi-asset portfolio...")
    portfolio_df, weights_df = run_portfolio(asset_results,
                                             transaction_cost_pct=BACKTEST_CFG["transaction_cost_pct"])
    pm = compute_metrics(portfolio_df, returns_col="portfolio_net", equity_col="equity",
                         label="Portfolio (Inv-Vol)")
    if verbose:
        print_metrics(pm)
    if save_plots:
        plot_portfolio(portfolio_df, weights_df, asset_results)
    if save_results:
        portfolio_df.to_csv("results/portfolio_equity.csv")

    # ── 7. Walk-forward ──
    print("\n[5] Walk-forward validation (EURUSD)...")
    wf_results = pd.DataFrame()
    if "EURUSD" in raw_dfs:
        wf_results = run_walk_forward_optimization(raw_dfs["EURUSD"], asset="EURUSD", n_splits=5)
        if not wf_results.empty and verbose:
            print(f"\n  OOS results ({len(wf_results)} folds):")
            print(wf_results[["fold", "window", "num_std", "regime", "sharpe", "cagr", "max_dd"]].to_string(index=False))
            print(f"\n  Median OOS Sharpe : {wf_results['sharpe'].median():.3f}")
            print(f"  Positive folds    : {(wf_results['sharpe'] > 0).sum()}/{len(wf_results)}")
        if save_results:
            wf_results.to_csv("results/wf_EURUSD.csv", index=False)

    # ── 8. Monte Carlo ──
    print("\n[6] Monte Carlo simulation (EURUSD)...")
    mc_results = {}
    if "EURUSD" in asset_results:
        mc_results = monte_carlo(asset_results["EURUSD"]["net_returns"], n_simulations=5000, label="EURUSD")
        if verbose:
            print_monte_carlo(mc_results)
        if save_plots:
            plot_monte_carlo(mc_results)

    # ── Summary ──
    print("\n" + "═" * 62)
    print("  SUMMARY — ALL ASSETS")
    print("═" * 62)
    print(f"\n  {'Asset':<12} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>10} {'Trades':>8}")
    print(f"  {'─'*50}")
    for n, m in asset_metrics.items():
        if "error" not in m:
            print(f"  {n:<12} {m['sharpe']:>8.3f} {m['cagr']:>8.2f} {m['max_drawdown']:>10.2f} {m['n_trades']:>8}")
    if "error" not in pm:
        print(f"  {'PORTFOLIO':<12} {pm['sharpe']:>8.3f} {pm['cagr']:>8.2f} {pm['max_drawdown']:>10.2f}")

    print(f"\n  Results: results/  |  Plots: results/plots/")
    print("═" * 62 + "\n")

    return dict(
        asset_results=asset_results, asset_metrics=asset_metrics,
        portfolio_df=portfolio_df, portfolio_metrics=pm,
        options_results=opt_results, wf_results=wf_results, mc_results=mc_results,
    )


# ════════════════════════════════════════════════════════════════
# SECTION 14 — CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRP Strategy — Master Runner")
    parser.add_argument("--optimize", action="store_true",
                        help="Run walk-forward parameter grid search")
    parser.add_argument("--ibkr",     action="store_true",
                        help="Fetch fresh data from Interactive Brokers")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    if args.ibkr:
        print("Fetching IBKR data...")
        for name in ["EURUSD", "USDJPY"]:
            try:
                fetch_ibkr_spot(name)
            except Exception as e:
                print(f"  IBKR {name} failed: {e}")
        try:
            fetch_ibkr_6e_continuous()
        except Exception as e:
            print(f"  IBKR 6E failed: {e}")

    if args.optimize:
        print("\nRunning optimization on all assets...")
        raw = load_all_assets()
        for name, df in raw.items():
            print(f"\n── {name} ──")
            top = full_grid_search(df).head(5)
            print("  Top 5 (in-sample):")
            print(top[["window", "num_std", "regime", "sharpe", "cagr", "max_dd"]].to_string(index=False))
            wf = run_walk_forward_optimization(df, asset=name)
            if not wf.empty:
                print(f"\n  Walk-forward OOS:")
                print(wf[["fold", "window", "num_std", "regime", "sharpe"]].to_string(index=False))
                print(f"  Median OOS Sharpe: {wf['sharpe'].median():.3f}")
    else:
        run_full_pipeline(save_plots=not args.no_plots)
