import pandas as pd
import numpy as np
from scipy.stats import linregress

def regression_slope(series, window=20):
    """Returns the slope of a linear regression line over a moving window."""
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series[i - window:i]
        x = np.arange(window)
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def bollinger_bands(series, window=20, num_std=2):
    """Returns upper and lower Bollinger Bands."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper = rolling_mean + (num_std * rolling_std)
    lower = rolling_mean - (num_std * rolling_std)
    return rolling_mean, upper, lower
