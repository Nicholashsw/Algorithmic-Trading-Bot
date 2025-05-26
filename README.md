# Trading Bot: Mean Reversion Strategy (Spot vs Futures)

This project explores a **mean-reversion trading strategy** applied to:

- 📈 **Spot EUR/USD** (from IBKR)
- 📉 **6E Futures** (continuous front-month, from proxy CSV)

We compare performance across both markets using a consistent strategy framework.

---

## 🧠 Strategy Overview

- Uses **Bollinger Bands** and **slope regression**
- Generates signals:
  - 🔴 **Short** when price > upper band and slope < 0
  - 🟢 **Long** when price < lower band and slope > 0

---

## 🔍 Key Features

- 20 years of backtested data (spot & futures)
- Visualizations: signal chart + equity curve
- Performance metrics:
  - Total Return
  - CAGR
  - Sharpe Ratio
  - Max Drawdown

---

## 📊 Sample Output

| Pair         | Total Return | CAGR   | Sharpe | Max Drawdown |
|--------------|--------------|--------|--------|---------------|
| EUR/USD Spot | -8.72%       | -0.46% | -0.29  | 15.44%        |
| 6E Futures   | 0.32%        | 0.02%  | 0.04   | 1.69%         |

---

## 📁 Project Structure

```bash
├── data/                       # Historical CSVs
├── utils/                     # Backtest engine + indicators
├── strategies/                # Mean reversion logic
├── backtest.py                # CLI runner
├── spot_vs_futures_strategy_analysis.ipynb  # Notebook visualization
├── fetch_6e_proxy.py          # External fetcher for 6E data
```

---

## ✅ Next Steps

- Improve position sizing and capital allocation
- Introduce volatility filters (e.g. ATR)
- Test other FX and futures pairs (JPY, GBP, AUD, etc.)
- Apply to intraday bars or resample hourly data

## Author

Nicholas Hong | Built for educational and research purposes. Not financial advice.


