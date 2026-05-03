"""
Parameter Optimization — Walk-Forward Grid Search
===================================================
Finds optimal BB parameters using walk-forward to prevent overfitting.
Tests across all 4 assets to find robust parameter set.

Usage: python optimize.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import itertools

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.indicators import bollinger_bands
from utils.backtest_engine import simulate_trades, compute_metrics

ASSETS = {
    "EURUSD": "data/EURUSD_yf.csv",
    "USDJPY": "data/USDJPY_yf.csv",
    "GBPUSD": "data/GBPUSD_yf.csv",
    "AUDUSD": "data/AUDUSD_yf.csv",
}

PARAM_GRID = {
    "window":   [10, 15, 20, 25, 30],
    "num_std":  [1.5, 2.0, 2.5],
    "regime":   ["none", "trend_aware"],
}

TC = 0.0001  # transaction cost


def load_df(path):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    return df.dropna(subset=["close"]).sort_index()


def run_params(df, window, num_std, regime, tc=TC):
    """Quick backtest for a parameter combo."""
    df = df.copy()
    bb_mean, upper, lower = bollinger_bands(df["close"], window=window, num_std=num_std)
    ma200 = df["close"].rolling(200).mean()

    if regime == "trend_aware":
        long_cond  = (df["close"] < lower) & (df["close"] > ma200)
        short_cond = (df["close"] > upper) & (df["close"] < ma200)
    else:
        long_cond  = df["close"] < lower
        short_cond = df["close"] > upper

    df["signal"] = 0
    df.loc[long_cond,  "signal"] = 1
    df.loc[short_cond, "signal"] = -1
    df["position"] = df["signal"].shift(1).fillna(0)
    df["returns"]  = df["close"].pct_change()
    df["strat_ret"] = df["returns"] * df["position"]
    df["costs"]    = df["position"].diff().abs() * tc
    df["net_ret"]  = df["strat_ret"] - df["costs"]
    df["equity"]   = (1 + df["net_ret"]).cumprod() * 100

    ret = df["net_ret"].dropna()
    if len(ret) < 100 or ret.std() == 0:
        return None
    sharpe  = ret.mean() / ret.std() * np.sqrt(252)
    total_r = df["equity"].iloc[-1] / 100 - 1
    n_yrs   = (df.index[-1] - df.index[0]).days / 365.25
    cagr    = (df["equity"].iloc[-1] / 100) ** (1/n_yrs) - 1 if n_yrs > 0 else 0
    max_dd  = ((df["equity"].cummax() - df["equity"]) / df["equity"].cummax()).max()
    n_trades = (df["position"].diff().abs() > 0).sum()
    return dict(sharpe=sharpe, cagr=cagr*100, total_ret=total_r*100,
                max_dd=max_dd*100, n_trades=int(n_trades))


def walk_forward_optimize(df, asset_name):
    """
    5-fold walk-forward: optimize on train, evaluate on test.
    Returns best params and OOS metrics.
    """
    n = len(df)
    fold_size = n // 5
    oos_records = []

    combos = list(itertools.product(
        PARAM_GRID["window"],
        PARAM_GRID["num_std"],
        PARAM_GRID["regime"],
    ))

    for fold in range(5):
        train_end = fold_size * (fold + 1)
        test_end  = min(train_end + fold_size // 2, n)
        if train_end >= n - 50:
            break

        train = df.iloc[:train_end]
        test  = df.iloc[train_end:test_end]
        if len(test) < 50:
            continue

        # Find best params on train set
        best_sharpe, best_params = -999, None
        for window, num_std, regime in combos:
            res = run_params(train, window, num_std, regime)
            if res and res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_params = (window, num_std, regime)

        if best_params is None:
            continue

        # Evaluate on test set
        w, s, r = best_params
        oos = run_params(test, w, s, r)
        if oos:
            oos["fold"]    = fold + 1
            oos["window"]  = w
            oos["num_std"] = s
            oos["regime"]  = r
            oos["train_sharpe"] = round(best_sharpe, 3)
            oos_records.append(oos)

    return pd.DataFrame(oos_records)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PARAMETER OPTIMIZATION — WALK-FORWARD GRID SEARCH")
    print("="*60)

    os.makedirs("results", exist_ok=True)
    all_summary = {}

    for asset, path in ASSETS.items():
        if not os.path.exists(path):
            continue
        df = load_df(path)
        print(f"\n── {asset} ({len(df)} rows) ──")

        # Full-period grid search (for reference only — not used for trading)
        print("  Full-period grid search:")
        records = []
        for window, num_std, regime in itertools.product(
            PARAM_GRID["window"], PARAM_GRID["num_std"], PARAM_GRID["regime"]
        ):
            res = run_params(df, window, num_std, regime)
            if res:
                res.update(window=window, num_std=num_std, regime=regime, asset=asset)
                records.append(res)

        if records:
            full_df = pd.DataFrame(records).sort_values("sharpe", ascending=False)
            top = full_df.head(5)
            print(top[["window","num_std","regime","sharpe","cagr","max_dd","n_trades"]].to_string(index=False))

        # Walk-forward (this is what matters)
        print(f"\n  Walk-forward OOS results:")
        wf = walk_forward_optimize(df, asset)
        if not wf.empty:
            print(wf[["fold","window","num_std","regime","sharpe","cagr","max_dd"]].to_string(index=False))
            print(f"\n  OOS Median Sharpe : {wf['sharpe'].median():.3f}")
            print(f"  OOS Min Sharpe    : {wf['sharpe'].min():.3f}")
            print(f"  Positive folds    : {(wf['sharpe']>0).sum()}/{len(wf)}")

            # Most common best params
            mode_params = wf[["window","num_std","regime"]].mode().iloc[0]
            print(f"  Most common best  : window={mode_params['window']}, "
                  f"std={mode_params['num_std']}, regime={mode_params['regime']}")
            all_summary[asset] = wf
            wf.to_csv(f"results/wf_{asset}.csv", index=False)

    # Cross-asset robust params: find params that work across ALL assets
    print("\n" + "="*60)
    print("  CROSS-ASSET ROBUSTNESS CHECK")
    print("="*60)

    # Test fixed params across all assets
    target_params = [(20, 2.0, "none"), (20, 2.0, "trend_aware"),
                     (15, 2.0, "none"), (25, 2.0, "none")]

    print(f"\n  {'Params':<30} {'EUR':>8} {'JPY':>8} {'GBP':>8} {'AUD':>8} {'Avg':>8}")
    print(f"  {'─'*70}")

    for window, num_std, regime in target_params:
        sharpes = []
        row = f"  w={window}, std={num_std}, {regime:<12}"
        for asset, path in ASSETS.items():
            if not os.path.exists(path):
                row += f"  {'N/A':>6}"
                continue
            df = load_df(path)
            res = run_params(df, window, num_std, regime)
            s = res["sharpe"] if res else 0
            sharpes.append(s)
            row += f"  {s:>6.3f}"
        avg = np.mean(sharpes) if sharpes else 0
        row += f"  {avg:>6.3f}"
        print(row)

    print(f"\n  Results saved to: results/wf_*.csv\n")
