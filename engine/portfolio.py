"""
Multi-Asset Portfolio Engine — v1.0
=====================================
Combines signals across multiple FX pairs using:
- Inverse volatility weighting (equal risk contribution)
- Correlation-aware diversification
- Portfolio-level metrics
"""

import pandas as pd
import numpy as np
from utils.backtest_engine import compute_metrics, print_metrics


def inverse_vol_weights(returns_dict, vol_window=63):
    """
    Compute inverse-volatility weights across assets.

    Parameters
    ----------
    returns_dict : dict of {asset_name: returns_series}
    vol_window   : rolling window for vol estimation

    Returns
    -------
    DataFrame of weights (index=dates, columns=asset names)
    """
    # Align all series on common dates
    returns_df = pd.DataFrame(returns_dict).dropna(how='all')

    # Rolling volatility
    vols = returns_df.rolling(vol_window).std() * np.sqrt(252)
    vols = vols.replace(0, np.nan)

    # Inverse vol weights (unnormalized)
    inv_vols = 1.0 / vols

    # Normalize so weights sum to 1
    weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)

    return weights


def run_portfolio(
    asset_dfs,
    transaction_cost_pct=0.0001,
    initial_capital=100.0,
    weighting="inv_vol",
    vol_window=63,
):
    """
    Run multi-asset portfolio backtest.

    Parameters
    ----------
    asset_dfs : dict of {asset_name: df_with_net_returns_column}
    transaction_cost_pct : cost per trade
    initial_capital : starting capital
    weighting : "inv_vol" | "equal"
    vol_window : vol estimation window for inv_vol

    Returns
    -------
    portfolio_df : DataFrame with portfolio equity and returns
    weights_df   : DataFrame with asset weights over time
    """
    # Collect net returns from each asset
    returns_dict = {}
    for name, df in asset_dfs.items():
        if "net_returns" in df.columns:
            returns_dict[name] = df["net_returns"]
        elif "strategy_returns" in df.columns:
            returns_dict[name] = df["strategy_returns"]

    if not returns_dict:
        raise ValueError("No strategy returns found in asset_dfs")

    returns_df = pd.DataFrame(returns_dict).fillna(0)

    # Compute weights
    if weighting == "inv_vol":
        weights_df = inverse_vol_weights(returns_dict, vol_window=vol_window)
        # Forward-fill weights (use previous day's weight for current day)
        weights_df = weights_df.shift(1).reindex(returns_df.index).fillna(
            1.0 / len(returns_dict)
        )
    else:
        # Equal weight
        weights_df = pd.DataFrame(
            1.0 / len(returns_dict),
            index=returns_df.index,
            columns=returns_df.columns
        )

    # Portfolio returns = weighted sum of asset returns
    port_returns = (returns_df * weights_df).sum(axis=1)

    # Transaction costs on weight changes
    weight_changes = weights_df.diff().abs().sum(axis=1)
    port_costs = weight_changes * transaction_cost_pct
    port_net_returns = port_returns - port_costs

    # Equity curve
    portfolio_df = pd.DataFrame({
        "portfolio_returns": port_returns,
        "portfolio_costs":   port_costs,
        "portfolio_net":     port_net_returns,
        "equity":            (1 + port_net_returns).cumprod() * initial_capital,
    })

    # Add individual asset net returns for attribution
    for name, ret in returns_dict.items():
        portfolio_df[f"ret_{name}"] = ret.reindex(portfolio_df.index).fillna(0)

    return portfolio_df, weights_df


def portfolio_correlation_matrix(asset_dfs):
    """Compute cross-asset return correlation matrix."""
    returns_dict = {}
    for name, df in asset_dfs.items():
        if "net_returns" in df.columns:
            returns_dict[name] = df["net_returns"]
    returns_df = pd.DataFrame(returns_dict).dropna()
    return returns_df.corr().round(3)


def rolling_sharpe(returns, window=252):
    """Compute rolling Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
