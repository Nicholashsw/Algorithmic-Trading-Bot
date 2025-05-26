import pandas as pd

def simulate_trades(df):
    df = df.copy()

    # Shift signal to create positions (enter on next day)
    df['position'] = df['signal'].shift(1).fillna(0)

    # Daily returns
    df['returns'] = df['close'].pct_change()

    # Strategy returns = returns * position
    df['strategy_returns'] = df['returns'] * df['position']

    # Equity curve = cumulative return on $100
    df['equity'] = (1 + df['strategy_returns']).cumprod() * 100

    return df
