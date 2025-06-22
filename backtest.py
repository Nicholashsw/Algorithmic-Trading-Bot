import pandas as pd
import matplotlib.pyplot as plt

from strategies.mean_reversion import apply_mean_reversion_strategy
from utils.backtest_engine import simulate_trades

# === Load 20Y EUR/USD Data ===
df = pd.read_csv("data/EURUSD_IBKR.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# === Apply Strategy ===
df = apply_mean_reversion_strategy(df, window=8, slope_window=3)

# === Count Trades ===
long_signals = (df["signal"] == 1).sum()
short_signals = (df["signal"] == -1).sum()
print(f"Long signals: {long_signals}")
print(f"Short signals: {short_signals}")

# === Run Backtest ===
df = simulate_trades(df)

# === Performance Metrics ===
total_return = df['equity'].iloc[-1] / 100 - 1
num_years = (df.index[-1] - df.index[0]).days / 365.25
cagr = (df['equity'].iloc[-1] / 100) ** (1 / num_years) - 1
sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * (252 ** 0.5)
max_dd = ((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max()

print("\n📊 Strategy Performance:")
print(f"• Total Return: {total_return:.2%}")
print(f"• CAGR: {cagr:.2%}")
print(f"• Sharpe Ratio: {sharpe:.2f}")
print(f"• Max Drawdown: {max_dd:.2%}")

# === Plot 1: Signals ===
plt.figure(figsize=(12, 6))
plt.plot(df["close"], label="Close Price", color="blue", linewidth=1)
plt.plot(df["upper"], linestyle="--", color="gray", alpha=0.9, linewidth=1, label="Upper Band")
plt.plot(df["lower"], linestyle="--", color="gray", alpha=0.9, linewidth=1, label="Lower Band")
plt.plot(df[df["signal"] == 1].index, df[df["signal"] == 1]["close"], "^", markersize=8, color="green", label="Long Signal")
plt.plot(df[df["signal"] == -1].index, df[df["signal"] == -1]["close"], "v", markersize=8, color="red", label="Short Signal")
plt.title("Mean Reversion Strategy Signals")
plt.legend()
plt.tight_layout()
plt.show()

# === Plot 2: Equity Curve ===
plt.figure(figsize=(12, 4))
plt.plot(df["equity"], color="orange", label="Equity Curve")
plt.title("Strategy Equity Curve")
plt.ylabel("Portfolio Value ($)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()
