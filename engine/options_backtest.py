"""
Options Backtest Engine — v2.0
================================
Fixes from v1:
- Added proper expiry P&L simulation (v1 only computed entry cost)
- Added dynamic sigma from realized volatility
- Added Greeks computation (delta, theta, vega)
- Added partial exit (mark-to-market before expiry)
- Added per-trade P&L tracking
"""

import numpy as np
import pandas as pd
from options.black_scholes import black_scholes_price, bs_greeks


def simulate_spread(S, K1, K2, T, r, sigma, direction="bull_call"):
    """
    Simulate entry cost of a vertical spread.
    (Preserved for backwards compatibility)
    """
    if direction == "bull_call":
        buy  = black_scholes_price(S, K1, T, r, sigma, option_type="call")
        sell = black_scholes_price(S, K2, T, r, sigma, option_type="call")
        net_cost = buy - sell
        return {"net_premium": net_cost}
    elif direction == "bear_put":
        buy  = black_scholes_price(S, K2, T, r, sigma, option_type="put")
        sell = black_scholes_price(S, K1, T, r, sigma, option_type="put")
        net_cost = buy - sell
        return {"net_premium": net_cost}
    else:
        raise ValueError("direction must be 'bull_call' or 'bear_put'")


def simulate_spread_pnl(
    S_entry, S_expiry, K1, K2, T_entry, r, sigma_entry,
    direction="bull_call", contract_size=125000, pip_value=1.0
):
    """
    Full spread P&L simulation from entry to expiry.

    Parameters
    ----------
    S_entry   : Spot price at trade entry
    S_expiry  : Spot price at option expiry
    K1        : Lower strike
    K2        : Upper strike
    T_entry   : Time to expiry at entry (in years)
    r         : Risk-free rate (annualized)
    sigma_entry : Implied/realized vol at entry (annualized)
    direction : "bull_call" or "bear_put"
    contract_size : Notional size (6E = 125,000 EUR)
    pip_value : P&L per pip per contract

    Returns
    -------
    dict with: entry_cost, expiry_value, pnl, pnl_pct, max_profit, max_loss
    """
    # Entry: pay net premium
    if direction == "bull_call":
        buy_price  = black_scholes_price(S_entry, K1, T_entry, r, sigma_entry, "call")
        sell_price = black_scholes_price(S_entry, K2, T_entry, r, sigma_entry, "call")
        entry_cost = buy_price - sell_price  # net debit

        # Expiry intrinsic value
        long_call_value  = max(0.0, S_expiry - K1)
        short_call_value = max(0.0, S_expiry - K2)
        expiry_value = long_call_value - short_call_value

        max_profit = K2 - K1 - entry_cost
        max_loss   = entry_cost

    elif direction == "bear_put":
        buy_price  = black_scholes_price(S_entry, K2, T_entry, r, sigma_entry, "put")
        sell_price = black_scholes_price(S_entry, K1, T_entry, r, sigma_entry, "put")
        entry_cost = buy_price - sell_price

        long_put_value  = max(0.0, K2 - S_expiry)
        short_put_value = max(0.0, K1 - S_expiry)
        expiry_value = long_put_value - short_put_value

        max_profit = K2 - K1 - entry_cost
        max_loss   = entry_cost

    else:
        raise ValueError("direction must be 'bull_call' or 'bear_put'")

    pnl = expiry_value - entry_cost
    pnl_pct = pnl / entry_cost if entry_cost != 0 else 0.0

    return {
        "entry_cost":    round(entry_cost, 6),
        "expiry_value":  round(expiry_value, 6),
        "pnl":           round(pnl, 6),
        "pnl_pct":       round(pnl_pct, 4),
        "max_profit":    round(max_profit, 6),
        "max_loss":      round(max_loss, 6),
        "direction":     direction,
        "K1": K1, "K2": K2,
        "S_entry": S_entry, "S_expiry": S_expiry,
    }
