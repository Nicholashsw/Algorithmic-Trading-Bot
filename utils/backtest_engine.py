"""
Backtest Engine — v2.0
=======================
Changes from v1:
- Added transaction costs (bid-ask spread + slippage)
- Added dynamic position sizing from signal_strength
- Added full performance metrics as a dict
- Added walk-forward split utility
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────

def simulate_trades(
    df,
    transaction_cost_pct=0.0001,  # 1 pip on 1.0 price = 0.01% per trade
    use_position_sizing=False,
    initial_capital=100.0,
):
    """
    Vectorized trade simulation with transaction costs.

    Parameters
    ----------
    df : DataFrame with 'close', 'signal' columns (and optionally 'position_size')
    transaction_cost_pct : Cost per trade entry/exit as fraction of price
    use_position_sizing : If True, use 'position_size' column for variable sizing
    initial_capital : Starting portfolio value

    Returns
    -------
    df with columns: position, returns, strategy_returns, costs, net_returns, equity
    """
    df = df.copy()

    # Positions: shift signal by 1 (enter next bar after signal)
    if use_position_sizing and "position_size" in df.columns:
        df["position"] = df["position_size"].shift(1).fillna(0)
    else:
        df["position"] = df["signal"].shift(1).fillna(0)

    # Daily returns
    df["returns"] = df["close"].pct_change()

    # Gross strategy returns
    df["strategy_returns"] = df["returns"] * df["position"]

    # Transaction costs: fired when position changes
    df["position_change"] = df["position"].diff().abs()
    df["costs"] = df["position_change"] * transaction_cost_pct

    # Net returns after costs
    df["net_returns"] = df["strategy_returns"] - df["costs"]

    # Equity curve
    df["equity"] = (1 + df["net_returns"]).cumprod() * initial_capital

    return df


# ─────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────

def compute_metrics(df, returns_col="net_returns", equity_col="equity", label="Strategy"):
    """
    Compute comprehensive performance metrics.

    Returns dict with all key statistics.
    """
    ret = df[returns_col].dropna()
    equity = df[equity_col].dropna()

    if len(ret) < 20:
        return {"label": label, "error": "insufficient data"}

    initial = equity.iloc[0]
    final = equity.iloc[-1]
    total_return = final / initial - 1

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (final / initial) ** (1 / n_years) - 1 if n_years > 0 else 0

    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0

    # Sortino: downside deviation only
    downside = ret[ret < 0].std()
    sortino = ret.mean() / downside * np.sqrt(252) if downside > 0 else 0

    # Max drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (per-day)
    winning_days = (ret > 0).sum()
    total_days = (ret != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Trade count (position changes)
    if "position" in df.columns:
        trades = df["position"].diff().abs()
        n_trades = (trades > 0).sum()
    else:
        n_trades = 0

    # Best/worst day
    best_day = ret.max()
    worst_day = ret.min()

    # Volatility
    ann_vol = ret.std() * np.sqrt(252)

    # Skewness and kurtosis
    skew = ret.skew()
    kurt = ret.kurt()

    return {
        "label": label,
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "ann_volatility": round(ann_vol * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "n_trades": int(n_trades),
        "best_day": round(best_day * 100, 2),
        "worst_day": round(worst_day * 100, 2),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "years": round(n_years, 1),
        "final_equity": round(final, 2),
    }


def print_metrics(metrics):
    """Pretty-print performance metrics dict."""
    label = metrics.get("label", "Strategy")
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return
    print(f"  Total Return   : {metrics['total_return']:>8.2f}%")
    print(f"  CAGR           : {metrics['cagr']:>8.2f}%")
    print(f"  Sharpe         : {metrics['sharpe']:>8.3f}")
    print(f"  Sortino        : {metrics['sortino']:>8.3f}")
    print(f"  Calmar         : {metrics['calmar']:>8.3f}")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:>8.2f}%")
    print(f"  Ann Volatility : {metrics['ann_volatility']:>8.2f}%")
    print(f"  Win Rate       : {metrics['win_rate']:>8.1f}%")
    print(f"  No. of Trades  : {metrics['n_trades']:>8d}")
    print(f"  Skewness       : {metrics['skewness']:>8.3f}")
    print(f"  Kurtosis       : {metrics['kurtosis']:>8.3f}")
    print(f"  Period (years) : {metrics['years']:>8.1f}")
    print(f"  Final Equity   : ${metrics['final_equity']:>8.2f}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# Walk-forward split
# ─────────────────────────────────────────────

def walk_forward_split(df, n_splits=5, train_ratio=0.7):
    """
    Generate walk-forward train/test splits.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    n_splits : number of splits
    train_ratio : fraction of each split used for training

    Yields
    ------
    (train_df, test_df, split_label) tuples
    """
    total = len(df)
    split_size = total // n_splits

    for i in range(n_splits):
        start = i * split_size
        end = start + split_size if i < n_splits - 1 else total
        chunk = df.iloc[start:end]
        n_train = int(len(chunk) * train_ratio)
        train = chunk.iloc[:n_train]
        test = chunk.iloc[n_train:]
        label = f"Split {i+1}: {train.index[0].date()} → {test.index[-1].date()}"
        if len(test) > 20:
            yield train, test, label
