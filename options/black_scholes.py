"""
Black-Scholes Model — v2.0
============================
New additions:
- bs_greeks(): compute delta, gamma, theta, vega, rho
- bs_iv(): implied vol from market price (Newton-Raphson)
- bs_price_surface(): vol surface simulation
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes Option Pricing.

    Parameters
    ----------
    S : Spot price
    K : Strike price
    T : Time to maturity in years
    r : Risk-free rate (annualized)
    sigma : Volatility (annualized)
    option_type : "call" or "put"
    """
    if T <= 0:
        if option_type == "call":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute all BS Greeks.

    Returns
    -------
    dict: delta, gamma, theta, vega, rho
    """
    if T <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega  = S * nd1 * np.sqrt(T) / 100  # per 1% change in vol

    if option_type == "call":
        delta = Nd1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * Nd2) / 365
        rho   = K * T * np.exp(-r * T) * Nd2 / 100
    else:
        delta = Nd1 - 1
        theta = (-(S * nd1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 6),  # per day
        "vega":  round(vega, 6),   # per 1% vol
        "rho":   round(rho, 6),
    }


def bs_iv(market_price, S, K, T, r, option_type="call"):
    """
    Compute implied volatility from a market price using Brent's method.

    Returns
    -------
    float : implied volatility, or np.nan if no solution found
    """
    if T <= 0 or market_price <= 0:
        return np.nan

    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price < intrinsic:
        return np.nan

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=200)
        return round(iv, 6)
    except (ValueError, RuntimeError):
        return np.nan
