"""
Technical Indicators — v2.0
============================
New additions:
- realized_volatility(): rolling annualized HV
- atr(): Average True Range
- regression_slope() speed improvement using numpy vectorized approach
- zscore(): rolling z-score
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress


def regression_slope(series, window=20):
    """
    Rolling regression slope over a window.
    Returns slope in same units as series per bar.
    Uses vectorized numpy for speed on large series.
    """
    arr = series.values.astype(float)
    n = len(arr)
    slopes = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, n):
        y = arr[i - window + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        slopes[i] = ((x - x_mean) * (y - y.mean())).sum() / x_var
    return pd.Series(slopes, index=series.index)


def bollinger_bands(series, window=20, num_std=2):
    """Upper and lower Bollinger Bands."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return rolling_mean, upper, lower


def realized_volatility(returns, window=21, annualize=True):
    """
    Rolling realized volatility (historical volatility).

    Parameters
    ----------
    returns : Series of daily returns (pct_change)
    window  : lookback in trading days (21 ≈ 1 month)
    annualize : if True, multiply by sqrt(252)

    Returns
    -------
    Series of annualized realized vol
    """
    rv = returns.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


def atr(df, window=14):
    """
    Average True Range — measures price volatility range.
    Expects df with columns: high, low, close
    """
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def zscore(series, window=20):
    """Rolling z-score of a series."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return (series - mean) / std


def rolling_correlation(s1, s2, window=60):
    """Rolling Pearson correlation between two series."""
    return s1.rolling(window).corr(s2)
