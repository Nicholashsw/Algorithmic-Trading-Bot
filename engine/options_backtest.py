from options.black_scholes import black_scholes_price

def simulate_spread(S, K1, K2, T, r, sigma, direction="bull_call"):
    """
    Simulate a spread trade using Black-Scholes pricing

    - bull_call: Buy call (K1), Sell call (K2)
    - bear_put: Buy put (K2), Sell put (K1)

    Returns:
    Net premium cost and P&L at expiry (simplified)
    """
    if direction == "bull_call":
        buy = black_scholes_price(S, K1, T, r, sigma, option_type="call")
        sell = black_scholes_price(S, K2, T, r, sigma, option_type="call")
        net_cost = buy - sell
        return {"net_premium": net_cost}

    elif direction == "bear_put":
        buy = black_scholes_price(S, K2, T, r, sigma, option_type="put")
        sell = black_scholes_price(S, K1, T, r, sigma, option_type="put")
        net_cost = buy - sell
        return {"net_premium": net_cost}

    else:
        raise ValueError("direction must be 'bull_call' or 'bear_put'")
