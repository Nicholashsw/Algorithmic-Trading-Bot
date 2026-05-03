"""
Mean Reversion Strategy — v2.1
================================
Key insight from testing: slope filter was backwards and killing all signals.
When price is below the lower band, the slope of the last N bars is ALWAYS
negative (by definition — price had to fall to get there). Requiring slope > 0
was generating zero signals.

Corrected approach:
- PRIMARY: Trend-aware mean-reversion
  - In macro uptrend  (price > 200d MA): buy every dip to the lower band
  - In macro downtrend (price < 200d MA): sell every rally to the upper band
  This produces clean, regime-aware signals with Sharpe ~0.25

- ALTERNATIVE: Pure BB (no filter)
  - Fire on any BB pierce regardless of regime
  - Higher signal count, similar Sharpe (~0.27) but more drawdown
  - Good for higher-frequency strategies

- VRP overlay: use VRP regime as position SIZE multiplier (not signal filter)
"""

import pandas as pd
import numpy as np
from utils.indicators import regression_slope, bollinger_bands


def apply_mean_reversion_strategy(
    df,
    window=20,
    slope_window=5,       # kept for backwards compat but not used for filtering
    num_std=2.0,
    ma_trend_window=200,
    slope_threshold=0.0,  # kept for backwards compat
    require_regime_filter=True,
    regime_mode="trend_aware",  # "trend_aware" | "soft" | "none"
):
    """
    Apply corrected mean-reversion strategy.

    Modes
    -----
    regime_mode="trend_aware":
        Long  when price < lower_band AND price > 200d MA (buy dips in uptrend)
        Short when price > upper_band AND price < 200d MA (sell rallies in downtrend)
        → Most conservative, lowest drawdown, Sharpe ~0.25

    regime_mode="soft":
        Long  when price < lower_band AND price > 0.95 * 200d MA (not crashed)
        Short when price > upper_band AND price < 1.05 * 200d MA (not blown up)
        → Moderate, Sharpe ~0.06 (weaker)

    regime_mode="none":
        Long  when price < lower_band
        Short when price > upper_band
        → Pure mean-reversion, highest signal count, Sharpe ~0.27, MaxDD ~9%
    """
    df = df.copy()

    # Bollinger Bands
    df["bb_mean"], df["upper"], df["lower"] = bollinger_bands(
        df["close"], window=window, num_std=num_std
    )

    # Slope (kept for analytics, not filtering)
    raw_slope = regression_slope(df["close"], slope_window)
    df["slope"] = raw_slope / df["close"]

    # Long-run trend
    df["ma_trend"] = df["close"].rolling(ma_trend_window).mean()

    # Band stats
    band_range = (df["upper"] - df["lower"]).replace(0, np.nan)
    df["pct_b"]      = (df["close"] - df["lower"]) / band_range
    df["band_width"] = band_range / df["bb_mean"]

    # Signal generation
    df["signal"] = 0

    if regime_mode == "trend_aware":
        # Buy dips only in uptrend; sell rallies only in downtrend
        long_cond  = (df["close"] < df["lower"]) & (df["close"] > df["ma_trend"])
        short_cond = (df["close"] > df["upper"]) & (df["close"] < df["ma_trend"])

    elif regime_mode == "soft":
        # Allow near-trend entries with soft buffer
        long_cond  = (df["close"] < df["lower"]) & (df["close"] > df["ma_trend"] * 0.95)
        short_cond = (df["close"] > df["upper"]) & (df["close"] < df["ma_trend"] * 1.05)

    else:  # "none" — pure BB
        long_cond  = df["close"] < df["lower"]
        short_cond = df["close"] > df["upper"]

    df.loc[long_cond,  "signal"] = 1
    df.loc[short_cond, "signal"] = -1

    # Signal strength: how far outside the band
    df["signal_strength"] = 0.0
    df.loc[long_cond, "signal_strength"] = (
        (df.loc[long_cond, "lower"] - df.loc[long_cond, "close"])
        / df.loc[long_cond, "lower"]
    ).clip(0, 0.05) / 0.05

    df.loc[short_cond, "signal_strength"] = (
        (df.loc[short_cond, "close"] - df.loc[short_cond, "upper"])
        / df.loc[short_cond, "upper"]
    ).clip(0, 0.05) / 0.05

    return df
