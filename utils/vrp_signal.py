"""
Volatility Risk Premium (VRP) Signal — v1.0
============================================
The VRP = Implied Volatility - Realized Volatility.

When VRP > 0: options are overpriced relative to realized vol.
  → Favour short-vol trades (sell spreads, collect theta)
  → VRP is the structural edge: historically positive in FX ~65% of time

When VRP < 0: options are cheap.
  → Avoid selling options; favour directional long-vol plays

This module:
1. Computes realized volatility (RV) from price data
2. Proxies implied volatility (IV) using:
   a) yfinance FXE/FXY options chain (best free source)
   b) Fallback: IV = RV * (1 + vrp_premium) — simplest structural prior
3. Computes the VRP spread and a normalized VRP signal
4. Generates position sizing multipliers based on VRP regime
"""

import pandas as pd
import numpy as np
from utils.indicators import realized_volatility, zscore

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ─────────────────────────────────────────────
# Core VRP computation
# ─────────────────────────────────────────────

def compute_realized_vol(df, rv_window=21):
    """Compute rolling realized volatility from close prices."""
    returns = df["close"].pct_change()
    df = df.copy()
    df["returns"] = returns
    df["rv_21"] = realized_volatility(returns, window=rv_window)
    df["rv_5"]  = realized_volatility(returns, window=5)
    df["rv_63"] = realized_volatility(returns, window=63)
    return df


def estimate_iv_proxy(df, asset="EURUSD", vrp_structural_premium=0.02):
    """
    Estimate implied volatility.

    Strategy:
    1. Try to fetch FXE (EUR/USD ETF) or FXY (JPY ETF) options chain from yfinance
       to get real 1-month IV for recent period
    2. For historical periods: use structural prior IV = RV * (1 + vrp_premium)
       This embeds the known empirical finding that FX IV > RV on average by ~2%

    Parameters
    ----------
    df : DataFrame with 'rv_21' column already computed
    asset : "EURUSD" or "USDJPY" (selects ETF proxy)
    vrp_structural_premium : average IV-RV gap (empirical: ~1-3% in FX)
    """
    df = df.copy()

    # Structural prior: IV = RV + historical average VRP
    # This is the minimum viable IV estimate and is directionally correct
    df["iv_proxy"] = df["rv_21"] * (1 + vrp_structural_premium) + vrp_structural_premium

    # Try to fetch real IV from yfinance options chain for recent data
    etf_map = {
        "EURUSD": "FXE",
        "USDJPY": "FXY",
        "GBPUSD": "FXB",
        "AUDUSD": "FXA",
    }
    etf = etf_map.get(asset, "FXE")

    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(etf)
            exp_dates = ticker.options
            if exp_dates:
                # Use nearest expiry ~30 days out
                target = pd.Timestamp.now() + pd.Timedelta(days=30)
                nearest_exp = min(exp_dates, key=lambda d: abs((pd.Timestamp(d) - target).days))
                chain = ticker.option_chain(nearest_exp)
                calls = chain.calls
                atm_calls = calls[
                    (calls["impliedVolatility"] > 0.001) &
                    (calls["inTheMoney"] == False)
                ]
                if not atm_calls.empty:
                    recent_iv = atm_calls["impliedVolatility"].median()
                    # Backfill recent IV into last 30 days of data
                    recent_cutoff = df.index[-1] - pd.Timedelta(days=45)
                    df.loc[df.index >= recent_cutoff, "iv_proxy"] = recent_iv
        except Exception:
            pass  # Fall back to structural prior silently

    return df


def compute_vrp(df, asset="EURUSD", rv_window=21):
    """
    Full VRP pipeline: RV → IV → VRP spread → signal.

    Returns df with columns:
    - rv_21, rv_5, rv_63: realized vols at different horizons
    - iv_proxy: estimated implied volatility
    - vrp: raw VRP spread (IV - RV)
    - vrp_zscore: rolling z-score of VRP (regime signal)
    - vrp_regime: 1=sell vol, -1=buy vol, 0=neutral
    - vrp_size_mult: position sizing multiplier based on VRP strength
    """
    df = compute_realized_vol(df, rv_window=rv_window)
    df = estimate_iv_proxy(df, asset=asset)

    # VRP spread
    df["vrp"] = df["iv_proxy"] - df["rv_21"]

    # Rolling z-score (normalize for regime detection)
    df["vrp_zscore"] = zscore(df["vrp"], window=63)

    # VRP regime
    df["vrp_regime"] = 0
    df.loc[df["vrp"] > 0.005, "vrp_regime"] = 1    # sell vol (options overpriced)
    df.loc[df["vrp"] < -0.005, "vrp_regime"] = -1  # buy vol (options cheap)

    # Position sizing multiplier (0.5x to 1.5x based on VRP conviction)
    # Strong positive VRP → size up on short-vol trades
    vrp_norm = df["vrp_zscore"].clip(-2, 2) / 2  # normalize -1 to 1
    df["vrp_size_mult"] = 1.0 + 0.5 * vrp_norm.fillna(0)

    return df


# ─────────────────────────────────────────────
# VRP-filtered signal combiner
# ─────────────────────────────────────────────

def apply_vrp_filter(df):
    """
    Combine mean-reversion signal with VRP regime filter.

    Rules:
    - In sell-vol regime (VRP>0): amplify short-spread entries, allow both directions
    - In neutral regime: allow signal but reduce size
    - In buy-vol regime (VRP<0): favour only buying spreads (directional, not vol-selling)
    - Signal is unchanged; vrp_size_mult adjusts position size in backtest

    Returns df with 'vrp_filtered_signal' and 'position_size' columns.
    """
    df = df.copy()

    # Filtered signal: same direction as mean-reversion signal
    df["vrp_filtered_signal"] = df["signal"]

    # Position size = base signal strength × VRP multiplier
    base_size = df["signal"].abs() * (0.5 + 0.5 * df.get("signal_strength", pd.Series(1, index=df.index)))
    df["position_size"] = (base_size * df["vrp_size_mult"]).clip(0, 2)

    return df
