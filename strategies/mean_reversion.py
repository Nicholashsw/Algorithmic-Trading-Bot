import pandas as pd
from utils.indicators import regression_slope, bollinger_bands

def apply_mean_reversion_strategy(df, window=8, slope_window=3):
    df = df.copy()

    # Calculate indicators
    df['slope'] = regression_slope(df['close'], slope_window)
    df['mean'], df['upper'], df['lower'] = bollinger_bands(df['close'], window)

    # RELAXED SIGNAL LOGIC — no slope for now
    df['signal'] = 0
    df.loc[df['close'] > df['upper'], 'signal'] = -1
    df.loc[df['close'] < df['lower'], 'signal'] = 1

    return df
