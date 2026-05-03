"""
Master Backtest Runner — v2.0
===============================
Runs the complete VRP strategy pipeline across all assets:
1. Load data
2. Apply mean-reversion strategy (v2: slope + regime filter)
3. Compute VRP signal (RV + IV proxy)
4. Run directional backtest with transaction costs
5. Run options strategy with expiry P&L
6. Combine into multi-asset portfolio
7. Walk-forward validation
8. Output all metrics

Usage:
    python run_backtest.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── Make sure imports work from project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.mean_reversion import apply_mean_reversion_strategy
from utils.vrp_signal import compute_vrp, apply_vrp_filter
from utils.backtest_engine import simulate_trades, compute_metrics, print_metrics, walk_forward_split
from engine.options_signal_runner import run_option_strategy
from engine.portfolio import run_portfolio, portfolio_correlation_matrix, rolling_sharpe

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ASSETS = {
    "EURUSD": {"file": "data/EURUSD_yf.csv",  "vrp_premium": 0.02},
    "USDJPY": {"file": "data/USDJPY_yf.csv",  "vrp_premium": 0.015},
    "GBPUSD": {"file": "data/GBPUSD_yf.csv",  "vrp_premium": 0.02},
    "AUDUSD": {"file": "data/AUDUSD_yf.csv",  "vrp_premium": 0.025},
}

STRATEGY_PARAMS = {
    "window":              20,
    "slope_window":        5,
    "num_std":             2.0,
    "ma_trend_window":     200,
    "slope_threshold":     0.0,
    "require_regime_filter": True,
    "regime_mode":         "none",   # "none"=pure BB, "trend_aware"=regime filtered
}

BACKTEST_PARAMS = {
    "transaction_cost_pct": 0.0001,
    "use_position_sizing":  True,
    "initial_capital":      100.0,
}

OPTIONS_PARAMS = {
    "r":               0.025,
    "T":               1/12,
    "otm_pct":         0.01,
    "use_dynamic_sigma": True,
    "apply_vrp_filter":  True,
    "holding_bars":      21,
}


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_asset(name, cfg):
    path = cfg["file"]
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: file not found at {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df[["open","high","low","close"]].copy()
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    df = df.sort_index()
    print(f"  Loaded {name}: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# Full pipeline for one asset
# ─────────────────────────────────────────────
def run_asset(name, df, cfg, verbose=True):
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Running: {name}")
        print(f"{'─'*55}")

    # 1. Mean-reversion strategy
    df = apply_mean_reversion_strategy(df, **STRATEGY_PARAMS)

    # 2. VRP signal
    df["returns"] = df["close"].pct_change()
    df = compute_vrp(df, asset=name)
    df = apply_vrp_filter(df)

    # 3. Directional backtest
    df = simulate_trades(df, **BACKTEST_PARAMS)

    # 4. Metrics
    metrics = compute_metrics(df, label=f"{name} Directional")
    if verbose:
        print_metrics(metrics)

    return df, metrics


# ─────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────
def run_walk_forward(name, df_full, n_splits=5):
    oos_results = []
    for train_df, test_df, label in walk_forward_split(df_full, n_splits=n_splits, train_ratio=0.7):
        # Fit: in this version we use fixed params (no optimization)
        # Apply strategy on test set only
        test_df = apply_mean_reversion_strategy(test_df, **STRATEGY_PARAMS)
        test_df["returns"] = test_df["close"].pct_change()
        test_df = compute_vrp(test_df, asset=name)
        test_df = apply_vrp_filter(test_df)
        test_df = simulate_trades(test_df, **BACKTEST_PARAMS)
        m = compute_metrics(test_df, label=label)
        m["split"] = label
        oos_results.append(m)

    oos_df = pd.DataFrame(oos_results)
    return oos_df


# ─────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────
def monte_carlo(returns, n_simulations=5000, label="Strategy"):
    arr = returns.dropna().values
    if len(arr) < 50:
        return None

    sharpes, cagrs, max_dds = [], [], []
    for _ in range(n_simulations):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        eq = (1 + sample).cumprod()
        sharpe = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
        cagr   = eq[-1] ** (252 / len(sample)) - 1
        dd     = ((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max()
        sharpes.append(sharpe)
        cagrs.append(cagr * 100)
        max_dds.append(dd * 100)

    pcts = [5, 25, 50, 75, 95]
    result = {
        "label": label,
        "sharpe_pct": {p: round(np.percentile(sharpes, p), 3) for p in pcts},
        "cagr_pct":   {p: round(np.percentile(cagrs, p), 2) for p in pcts},
        "maxdd_pct":  {p: round(np.percentile(max_dds, p), 2) for p in pcts},
    }
    return result


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_asset(name, df):
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    # 1. Price + Bollinger Bands + Signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["close"], color="#1f77b4", linewidth=0.7, label="Close")
    ax1.plot(df["upper"], "--", color="#aec7e8", linewidth=0.5, label="Upper Band")
    ax1.plot(df["lower"], "--", color="#aec7e8", linewidth=0.5, label="Lower Band")
    ax1.plot(df["bb_mean"], ":", color="#6baed6", linewidth=0.5, label="BB Mean")
    ax1.plot(df["ma_trend"], color="#ff7f0e", linewidth=0.8, alpha=0.7, label="200d MA")
    longs  = df[df["signal"] == 1]
    shorts = df[df["signal"] == -1]
    ax1.scatter(longs.index,  longs["close"],  marker="^", color="green", s=15, zorder=5, label="Long")
    ax1.scatter(shorts.index, shorts["close"], marker="v", color="red",   s=15, zorder=5, label="Short")
    ax1.set_title(f"{name} — Price, Bollinger Bands & Signals", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7, ncol=4)
    ax1.set_ylabel("Price")

    # 2. Equity curve
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df["equity"], color="#2ca02c", linewidth=1.2, label="Strategy Equity")
    ax2.axhline(100, color="gray", linewidth=0.5, linestyle="--")
    ax2.fill_between(df.index, df["equity"], 100, where=df["equity"] < 100,
                     alpha=0.3, color="red", label="Underwater")
    ax2.set_title(f"{name} — Equity Curve (Transaction Costs Included)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend(fontsize=8)

    # 3. Drawdown
    ax3 = fig.add_subplot(gs[2, 0])
    rolling_max = df["equity"].cummax()
    drawdown = (df["equity"] - rolling_max) / rolling_max * 100
    ax3.fill_between(df.index, drawdown, 0, color="red", alpha=0.5)
    ax3.set_title("Drawdown (%)", fontsize=10)
    ax3.set_ylabel("%")

    # 4. VRP signal
    ax4 = fig.add_subplot(gs[2, 1])
    if "vrp" in df.columns:
        vrp_pct = df["vrp"] * 100
        ax4.plot(vrp_pct, color="#9467bd", linewidth=0.7)
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.fill_between(df.index, vrp_pct, 0, where=vrp_pct > 0, alpha=0.3, color="green", label="Sell Vol")
        ax4.fill_between(df.index, vrp_pct, 0, where=vrp_pct < 0, alpha=0.3, color="red", label="Buy Vol")
        ax4.set_title("VRP = IV - RV (%)", fontsize=10)
        ax4.set_ylabel("%")
        ax4.legend(fontsize=7)

    # 5. Rolling Sharpe
    ax5 = fig.add_subplot(gs[3, 0])
    roll_sharpe = rolling_sharpe(df["net_returns"].fillna(0), window=252)
    ax5.plot(roll_sharpe, color="#d62728", linewidth=0.8)
    ax5.axhline(0, color="black", linewidth=0.5)
    ax5.axhline(0.5, color="green", linewidth=0.5, linestyle="--", label="0.5 target")
    ax5.set_title("Rolling 1Y Sharpe", fontsize=10)
    ax5.set_ylabel("Sharpe")
    ax5.legend(fontsize=7)

    # 6. Signal distribution
    ax6 = fig.add_subplot(gs[3, 1])
    n_long  = (df["signal"] == 1).sum()
    n_short = (df["signal"] == -1).sum()
    n_flat  = (df["signal"] == 0).sum()
    bars = ax6.bar(["Long", "Short", "Flat"], [n_long, n_short, n_flat],
                   color=["green", "red", "gray"])
    ax6.bar_label(bars, fmt="%d", fontsize=8)
    ax6.set_title("Signal Distribution", fontsize=10)
    ax6.set_ylabel("Count")

    plt.suptitle(f"{name} VRP Strategy — Full Analysis", fontsize=13, fontweight="bold", y=1.01)
    path = f"results/plots/{name}_analysis.png"
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_portfolio(portfolio_df, weights_df, asset_names):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. Portfolio equity
    axes[0].plot(portfolio_df["equity"], color="#1f77b4", linewidth=1.5, label="Portfolio")
    for name in asset_names:
        col = f"ret_{name}"
        if col in portfolio_df.columns:
            eq = (1 + portfolio_df[col].fillna(0)).cumprod() * 100
            axes[0].plot(eq, linewidth=0.6, alpha=0.5, label=name)
    axes[0].axhline(100, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title("Multi-Asset Portfolio vs Individual Assets", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend(fontsize=8)

    # 2. Portfolio drawdown
    roll_max = portfolio_df["equity"].cummax()
    dd = (portfolio_df["equity"] - roll_max) / roll_max * 100
    axes[1].fill_between(portfolio_df.index, dd, 0, color="red", alpha=0.5)
    axes[1].set_title("Portfolio Drawdown (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("%")

    # 3. Asset weights over time
    weights_to_plot = weights_df.reindex(portfolio_df.index).fillna(0)
    weights_to_plot.plot.area(ax=axes[2], alpha=0.7, colormap="tab10")
    axes[2].set_title("Inverse-Vol Asset Weights Over Time", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Weight")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    path = "results/plots/portfolio_analysis.png"
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  VRP STRATEGY — FULL BACKTEST (v2.0)")
    print("="*60)

    # ── Load all assets ──
    print("\n[1] Loading Data...")
    asset_dfs_raw = {}
    for name, cfg in ASSETS.items():
        df = load_asset(name, cfg)
        if df is not None:
            asset_dfs_raw[name] = df

    # ── Run per-asset strategies ──
    print("\n[2] Running Per-Asset Strategies...")
    asset_results   = {}
    asset_metrics   = {}
    all_plot_paths  = []

    for name, df in asset_dfs_raw.items():
        df_result, metrics = run_asset(name, df, ASSETS[name], verbose=True)
        asset_results[name] = df_result
        asset_metrics[name] = metrics
        path = plot_asset(name, df_result)
        all_plot_paths.append(path)

    # ── Options strategy on EURUSD ──
    print("\n[3] Running Options Strategy (EURUSD)...")
    if "EURUSD" in asset_results:
        eurusd_df = asset_results["EURUSD"]
        options_results, options_equity = run_option_strategy(eurusd_df, **OPTIONS_PARAMS)

        if not options_results.empty:
            n_trades   = len(options_results)
            win_rate   = options_results["win"].mean() * 100
            avg_pnl    = options_results["pnl"].mean()
            total_pnl  = options_results["pnl"].sum()
            avg_ret    = options_results["pnl_pct"].mean() * 100

            print(f"\n  Options Strategy Summary:")
            print(f"  {'─'*40}")
            print(f"  Total Trades   : {n_trades}")
            print(f"  Win Rate       : {win_rate:.1f}%")
            print(f"  Avg P&L/trade  : {avg_pnl:.6f}")
            print(f"  Total P&L      : {total_pnl:.4f}")
            print(f"  Avg Return %   : {avg_ret:.2f}%")

            # Quick Sharpe on options P&L
            pnl_series = options_results["pnl"]
            if pnl_series.std() > 0:
                opt_sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(12)  # monthly
                print(f"  Sharpe (monthly): {opt_sharpe:.3f}")

            options_results.to_csv("results/options_trades.csv")
            print(f"  Saved: results/options_trades.csv")
        else:
            print("  No options trades generated (all filtered by VRP)")

    # ── Multi-asset portfolio ──
    print("\n[4] Building Multi-Asset Portfolio...")
    if len(asset_results) >= 2:
        portfolio_df, weights_df = run_portfolio(
            asset_results,
            transaction_cost_pct=0.0001,
            weighting="inv_vol",
            vol_window=63,
        )
        port_metrics = compute_metrics(
            portfolio_df,
            returns_col="portfolio_net",
            equity_col="equity",
            label="Multi-Asset Portfolio (Inv-Vol Weighted)"
        )
        print_metrics(port_metrics)

        plot_portfolio(portfolio_df, weights_df, list(asset_results.keys()))

        # Correlation matrix
        corr = portfolio_correlation_matrix(asset_results)
        print("\n  Strategy Return Correlations:")
        print(corr.to_string())

        portfolio_df.to_csv("results/portfolio_equity.csv")

    # ── Walk-forward validation ──
    print("\n[5] Walk-Forward Validation (EURUSD)...")
    if "EURUSD" in asset_dfs_raw:
        wf_results = run_walk_forward("EURUSD", asset_dfs_raw["EURUSD"], n_splits=5)
        oos_sharpe = wf_results["sharpe"].dropna()
        print(f"\n  Out-of-Sample Sharpe across {len(wf_results)} splits:")
        for _, row in wf_results.iterrows():
            if "error" not in row:
                print(f"    {row.get('split','?')[:60]}")
                print(f"      Sharpe: {row['sharpe']:.3f}  |  Return: {row['total_return']:.2f}%  |  MaxDD: {row['max_drawdown']:.2f}%")
        if len(oos_sharpe) > 0:
            print(f"\n  Median OOS Sharpe : {oos_sharpe.median():.3f}")
            print(f"  Min OOS Sharpe    : {oos_sharpe.min():.3f}")
            print(f"  Positive splits   : {(oos_sharpe > 0).sum()}/{len(oos_sharpe)}")
        wf_results.to_csv("results/walkforward_results.csv", index=False)

    # ── Monte Carlo ──
    print("\n[6] Monte Carlo Simulation (EURUSD)...")
    if "EURUSD" in asset_results:
        mc = monte_carlo(
            asset_results["EURUSD"]["net_returns"].dropna(),
            n_simulations=5000,
            label="EURUSD"
        )
        if mc:
            print(f"\n  Monte Carlo — 5,000 bootstrap simulations:")
            print(f"  {'Percentile':<12} {'Sharpe':>8} {'CAGR %':>8} {'MaxDD %':>8}")
            print(f"  {'─'*40}")
            for p in [5, 25, 50, 75, 95]:
                print(f"  {f'P{p}':<12} "
                      f"{mc['sharpe_pct'][p]:>8.3f} "
                      f"{mc['cagr_pct'][p]:>8.2f} "
                      f"{mc['maxdd_pct'][p]:>8.2f}")

    # ── Final summary ──
    print("\n" + "="*60)
    print("  FINAL SUMMARY — ALL ASSETS")
    print("="*60)
    print(f"\n  {'Asset':<12} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>10} {'WinRate%':>10} {'Trades':>8}")
    print(f"  {'─'*60}")
    for name, m in asset_metrics.items():
        if "error" not in m:
            print(f"  {name:<12} {m['sharpe']:>8.3f} {m['cagr']:>8.2f} {m['max_drawdown']:>10.2f} {m['win_rate']:>10.1f} {m['n_trades']:>8}")

    if "port_metrics" in dir() and "error" not in port_metrics:
        m = port_metrics
        print(f"  {'PORTFOLIO':<12} {m['sharpe']:>8.3f} {m['cagr']:>8.2f} {m['max_drawdown']:>10.2f} {m['win_rate']:>10.1f} {m['n_trades']:>8}")

    print(f"\n  Plots saved to: results/plots/")
    print(f"  Results saved to: results/")
    print("\n" + "="*60 + "\n")
