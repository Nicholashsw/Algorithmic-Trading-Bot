import pandas as pd
from utils.indicators import regression_slope, bollinger_bands

def apply_mean_reversion_strategy(df, window=20, slope_window=50):
    df = df.copy()

    # Calculate indicators
    df['slope'] = regression_slope(df['Close'], slope_window)
    df['mean'], df['upper'], df['lower'] = bollinger_bands(df['Close'], window)

    # Signal logic
    df['signal'] = 0

    # Condition 1: Downtrend & price at top → Sell bull call spread (short bias = -1)
    df.loc[(df['slope'] < 0) & (df['Close'] >= df['upper']), 'signal'] = -1

    # Condition 2: Uptrend & price at bottom → Sell bear put spread (long bias = +1)
    df.loc[(df['slope'] > 0) & (df['Close'] <= df['lower']), 'signal'] = 1

    return df

