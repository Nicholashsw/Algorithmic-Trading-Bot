import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes Option Pricing Formula

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate (annualized)
    sigma: Volatility of underlying (annualized)
    option_type: "call" or "put"

    Returns:
    Option price (float)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
