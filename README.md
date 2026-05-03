# 📘 Volatility Risk Premium — Multi-Asset FX Mean-Reversion Strategy

**Author**: Nicholas Hong | nhong001@e.ntu.edu.sg
**Mentored by**: Jirong Huang, Quantitative Portfolio Manager, Eastspring Investments
**Status**: v2.0 — Backtested & Validated | Paper Trading Next

> This repository implements a systematic mean-reversion strategy across FX pairs, overlaid with a Volatility Risk Premium (VRP) signal for dynamic position sizing and options entry filtering.

---

## 🔍 Strategy Overview

The strategy captures the tendency of FX rates to mean-revert after brief excursions outside Bollinger Bands. VRP (IV − Realized Vol) is used to:
- **Size positions**: larger when vol is overpriced (sell-vol regime)
- **Filter options entries**: only sell spreads when IV > RV (structural edge)

| Component | Detail |
|---|---|
| Signal | Bollinger Band pierce (price outside ±σ bands) |
| Regime filter | 200d MA direction (asset-specific, optional) |
| VRP signal | Rolling RV vs IV proxy; VRP z-score for sizing |
| Options | Bull-call / bear-put spreads with BSM pricing + expiry P&L |
| Execution | 1-pip transaction cost per trade (simulated) |
| Universe | EUR/USD, USD/JPY, GBP/USD, AUD/USD |

---

## 📊 v2.0 Results (Real 20Y Data, 2004–2025)

> **v1 results were on synthetic data. These are the real numbers.**

| Asset | Sharpe | CAGR | Total Return | Max Drawdown | Trades |
|---|---|---|---|---|---|
| EUR/USD | **0.60** | 3.6% | +118.9% | -11.9% | 1,531 |
| GBP/USD | **0.47** | 1.3% | +31.8% | -7.0% | 626 |
| AUD/USD | **0.31** | 1.0% | +21.7% | -8.6% | 424 |
| USD/JPY | 0.17 | 0.4% | +8.5% | -4.2% | 188 |
| **Portfolio (Inv-Vol)** | **0.41** | 1.0% | +24.6% | -8.9% | — |

### Robustness Tests

**Monte Carlo (EURUSD, 5,000 bootstrap simulations)**:
- P5 Sharpe: **0.31** — worst-case still positive ✅
- P50 Sharpe: **0.60** — confirms full-period results
- P95 Sharpe: **0.87** — upside scenario

**Walk-Forward Validation (EURUSD, out-of-sample only)**:
- Median OOS Sharpe: **0.79**
- Min OOS Sharpe: **0.44**
- Positive folds: **4/4** — no deterioration out-of-sample ✅

**Cross-Asset Correlation** (strategy returns): 0.03–0.14 → very low, strong diversification ✅

---

## 📁 Project Structure

```
trading-bot/
├── data/                          # FX data (yfinance + IBKR)
│   ├── EURUSD_yf.csv              # 2004–2025, 5706 rows
│   ├── USDJPY_yf.csv              # 2004–2025
│   ├── GBPUSD_yf.csv              # 2004–2025
│   ├── AUDUSD_yf.csv              # 2006–2025
│   └── EURUSD_IBKR.csv            # IBKR 20Y spot (backup)
├── strategies/
│   └── mean_reversion.py          # v2: fixed slope filter, regime modes
├── utils/
│   ├── indicators.py              # BB, regression slope, RV, ATR, zscore
│   ├── vrp_signal.py              # VRP pipeline: RV→IV→spread→signal
│   └── backtest_engine.py         # v2: TC, position sizing, metrics, walk-forward
├── engine/
│   ├── options_backtest.py        # v2: full expiry P&L simulation
│   ├── options_signal_runner.py   # v2: dynamic sigma, VRP filter, trade log
│   └── portfolio.py               # Multi-asset inv-vol weighting
├── options/
│   └── black_scholes.py           # v2: Greeks, implied vol (Newton-Raphson)
├── fetchers/                      # IBKR + yfinance data fetchers
├── results/                       # Backtest outputs, walk-forward CSVs
│   └── plots/                     # All charts
├── run_backtest.py                # Master backtest runner
├── optimize.py                    # Walk-forward parameter grid search
└── vrp_strategy_analysis_v2.ipynb # Comprehensive analysis notebook
```

---

## 🐛 v1 → v2 Bug Fixes

| Issue | v1 | v2 |
|---|---|---|
| Slope filter | Disabled as "RELAXED LOGIC" — was generating 0 signals | Removed; replaced with regime filter |
| Sigma | Hardcoded `sigma=0.12` | Rolling realized vol (21-day HV) |
| Options P&L | Only computed entry premium | Full expiry P&L with intrinsic value |
| Data | 31-row synthetic EURUSD.csv | 5,706 rows real data (2004–2025) |
| Assets | EUR/USD only | EUR/USD + USD/JPY + GBP/USD + AUD/USD |
| Portfolio | None | Inverse-volatility weighted multi-asset |
| Validation | None | 5-fold walk-forward + 5,000-sim Monte Carlo |

---

## 🧠 Key Insight: Why the Slope Filter Broke

In v1, the strategy required `slope > 0` for long entries. But when price is **below** the lower Bollinger Band (the entry condition), the 5-bar regression slope is **always negative** (price fell to get there). This generated **zero signals** on 20 years of real data.

**Fix**: Use the 200d MA as a regime indicator instead. The correct logic:
- **Trend-aware mode**: Long when price < lower BB AND price > 200d MA (buy dips in uptrends)
- **Pure BB mode**: Long when price < lower BB (any regime — works best for EUR/USD)

---

## ⚙️ Optimal Parameters (Walk-Forward Validated)

| Asset | Window | Std Dev | Regime Mode | OOS Sharpe |
|---|---|---|---|---|
| EUR/USD | 10 | 1.5 | none (pure BB) | 0.79 median |
| GBP/USD | 10 | 1.5 | trend_aware | 0.55 median |
| AUD/USD | 15 | 1.5 | trend_aware | 0.24 median |
| USD/JPY | 20 | 2.0 | trend_aware | ~0.00 (weak) |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib yfinance

# Fetch data (requires internet)
python fetchers/fetch_data.py     # EURUSD spot via yfinance
# Data already included in data/ folder

# Run full backtest
python run_backtest.py

# Run optimization (takes ~2 min)
python optimize.py

# Open analysis notebook
jupyter notebook vrp_strategy_analysis_v2.ipynb
```

---

## ✅ Milestones

- [x] 20Y real FX data pipeline (yfinance + IBKR)
- [x] Corrected mean-reversion signal (slope filter bug fixed)
- [x] VRP signal: rolling RV vs IV proxy
- [x] Options backtest v2: full expiry P&L simulation
- [x] Multi-asset portfolio: EUR/USD, GBP/USD, AUD/USD, USD/JPY
- [x] Walk-forward optimization (30-param grid, 5 folds)
- [x] Monte Carlo validation (5,000 simulations)
- [ ] Real IV data from IBKR/FXE options chain
- [ ] IBKR paper trading live signals
- [ ] Telegram alert bot for daily signals
- [ ] USD/CAD (6C) as 5th asset
- [ ] Intraday data + higher frequency testing

---

## 📣 Author

**Nicholas Hong** | NTU Singapore
Built for research and educational purposes. Not financial advice.

**Acknowledgments**: Jirong Huang, Quantitative Portfolio Manager at Eastspring Investments.
