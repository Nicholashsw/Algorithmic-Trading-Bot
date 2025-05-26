import pandas as pd
from options.black_scholes import black_scholes_price
from engine.options_backtest import simulate_spread

def run_option_strategy(df, r=0.01, sigma=0.12, T=0.25, otm_pct=0.02):
    """
    df: DataFrame with index as date and columns ['close', 'signal']
    r: risk-free rate
    sigma: volatility
    T: time to expiry in years
    otm_pct: strike distance for OTM legs
    """
    results = []

    for date, row in df.iterrows():
        signal = row['signal']
        spot = row['close']

        if signal == 1:
            # Bear Put: Buy higher K, sell lower K
            K1 = spot
            K2 = spot * (1 + otm_pct)
            trade = simulate_spread(spot, K1, K2, T, r, sigma, direction="bear_put")

        elif signal == -1:
            # Bull Call: Buy lower K, sell higher K
            K1 = spot * (1 - otm_pct)
            K2 = spot
            trade = simulate_spread(spot, K1, K2, T, r, sigma, direction="bull_call")

        else:
            continue

        results.append({
            "date": date,
            "spot": spot,
            "signal": signal,
            "K1": round(K1, 4),
            "K2": round(K2, 4),
            "net_premium": round(trade["net_premium"], 6)
        })

    return pd.DataFrame(results)
