"""
Options Signal Runner — v2.0
==============================
Fixes from v1:
- Uses dynamic sigma (realized volatility) instead of hardcoded 0.12
- Simulates actual expiry P&L (not just entry premium)
- Applies VRP filter: only sell spreads when VRP > 0
- Returns trade log with full per-trade P&L
- Computes options equity curve
"""

import pandas as pd
import numpy as np
from options.black_scholes import black_scholes_price
from engine.options_backtest import simulate_spread, simulate_spread_pnl


def run_option_strategy(
    df,
    r=0.025,             # risk-free rate (updated from 0.01 → ~2024 rate)
    T=1/12,              # time to expiry: 1 month (was 0.25 = 3 months)
    otm_pct=0.01,        # strike distance: 1% OTM (was 2%)
    use_dynamic_sigma=True,  # use RV column if available
    apply_vrp_filter=True,   # only sell when VRP > 0
    holding_bars=21,         # simulate holding to expiry (~1 month of trading days)
):
    """
    Run options spread strategy with proper expiry P&L.

    Parameters
    ----------
    df : DataFrame with 'close', 'signal', optionally 'rv_21', 'vrp'
    r  : risk-free rate
    T  : time to expiry in years
    otm_pct : OTM distance for spread legs
    use_dynamic_sigma : if True, use rv_21 column as sigma
    apply_vrp_filter  : if True, skip trades when VRP <= 0
    holding_bars      : trading days to simulate until expiry

    Returns
    -------
    tuple: (results_df, equity_series)
    """
    results = []
    price_arr = df["close"].values
    dates = df.index

    for i, (date, row) in enumerate(df.iterrows()):
        signal = row["signal"]
        spot = row["close"]

        # Skip no-signal days
        if signal == 0:
            continue

        # Dynamic sigma from realized vol
        if use_dynamic_sigma and "rv_21" in df.columns:
            sigma = row["rv_21"]
            if pd.isna(sigma) or sigma <= 0.005:
                sigma = 0.08  # fallback floor
        else:
            sigma = 0.08  # structural default for FX

        # VRP filter: only enter if VRP is positive (options overpriced)
        if apply_vrp_filter and "vrp" in df.columns:
            vrp = row.get("vrp", 0)
            if pd.isna(vrp) or vrp <= 0:
                continue  # skip: options are cheap, don't sell them

        # Strike selection
        if signal == 1:
            # Expect price to RISE → Bull Call Spread
            K1 = spot * (1 - otm_pct)   # buy slightly OTM call
            K2 = spot * (1 + otm_pct)   # sell further OTM call
            direction = "bull_call"

        elif signal == -1:
            # Expect price to FALL → Bear Put Spread
            K1 = spot                    # buy ATM put
            K2 = spot * (1 + otm_pct)   # buy higher strike put, sell K1
            # Correct: Bear Put = buy K2, sell K1 where K2 > K1
            K1_strike = spot * (1 - otm_pct)
            K2_strike = spot
            K1, K2 = K1_strike, K2_strike
            direction = "bear_put"

        # Find expiry bar
        expiry_idx = min(i + holding_bars, len(df) - 1)
        S_expiry = df["close"].iloc[expiry_idx]

        # Full P&L simulation
        trade = simulate_spread_pnl(
            S_entry=spot,
            S_expiry=S_expiry,
            K1=K1, K2=K2,
            T_entry=T,
            r=r,
            sigma_entry=sigma,
            direction=direction,
        )

        results.append({
            "entry_date":    date,
            "expiry_date":   dates[expiry_idx],
            "spot_entry":    round(spot, 5),
            "spot_expiry":   round(S_expiry, 5),
            "signal":        signal,
            "direction":     direction,
            "K1":            round(K1, 5),
            "K2":            round(K2, 5),
            "sigma_used":    round(sigma, 4),
            "entry_cost":    trade["entry_cost"],
            "expiry_value":  trade["expiry_value"],
            "pnl":           trade["pnl"],
            "pnl_pct":       trade["pnl_pct"],
            "max_profit":    trade["max_profit"],
            "max_loss":      trade["max_loss"],
            "win":           trade["pnl"] > 0,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return results_df, pd.Series(dtype=float)

    results_df = results_df.set_index("entry_date")

    # Build equity curve from options P&L
    # Each trade's P&L as fraction of entry cost → scale to $100 base
    results_df["cumulative_pnl"] = results_df["pnl"].cumsum()
    equity = 100 + results_df["cumulative_pnl"] / results_df["entry_cost"].mean() * 100
    results_df["equity"] = equity

    return results_df, equity
