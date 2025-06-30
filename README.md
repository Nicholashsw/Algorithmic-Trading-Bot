# Volatility Risk Premium & Risk Filter in Futures Options Strategy

This repository explores a **multi-market trading strategy** that combines:
- Mean-reversion signals from **Bollinger Bands + slope**
- Execution via **spot FX (EUR/USD)**, **futures (6E)**, and **options spreads**
- A core focus on **volatility risk premium (VRP)** and **signal-based filters** for directional trading

Designed and built for educational and research purposes by Nicholas Hong.

---

## 🔍 Project Focus
This project studies how systematic mean-reverting signals can be used to trade FX via:

- **Spot**: Direct exposure to FX prices (e.g., EUR/USD)
- **Futures**: 6E contract (Euro FX Futures)
- **Options**: Directional spreads exploiting VRP (bull call / bear put)

Backtests cover **20+ years** of EUR/USD and 6E price history. A later extension includes USD/JPY.

---

## Strategy Logic

### Mean-Reversion Signal
- Compute **regression slope** over short window to capture local trend
- Overlay **Bollinger Bands** (±2 std) for overbought/oversold detection
- Filter entry points based on slope direction:
  - Enter **long** at lower band if uptrend (slope > 0)
  - Enter **short** at upper band if downtrend (slope < 0)

### Options Spread Execution (New)
- On valid signals, enter **directional spreads**:
  - **Bull Call Spread** on downtrend oversold (expect reversal)
  - **Bear Put Spread** on uptrend overbought
- Pricing based on **Black-Scholes** approximation
- Simulated expiry P&L and strategy-level equity curve

---

## 📁 Folder Structure

```plaintext
trading-bot/
├── data/                          # Raw and cleaned CSVs (EURUSD, 6E, USDJPY)
├── engine/                        # Backtesting & signal runners
│   ├── backtest.py
│   ├── options_backtest.py
│   └── options_signal_runner.py
├── fetchers/                      # Scripts for data fetching (IBKR, proxies)
│   ├── fetch_data.py
│   ├── ibkr_fetch.py
│   ├── fetch_6e_continuous.py
│   ├── fetch_6e_proxy.py
│   ├── fetch_usdjpy_ibkr.py
│   └── fetch_6j_futures.py
├── strategies/                    # Mean-reversion logic
│   └── mean_reversion.py
├── utils/                         # Indicators, slope, engine
│   ├── indicators.py
│   └── backtest_engine.py
├── options/                       # Options pricing model
│   └── black_scholes.py
├── notebook_analysis_spot_vs_futures.ipynb
├── notebook_analysis_eurusd.ipynb
├── notebook_analysis_usdjpy.ipynb
├── README.md                      # You are here
└── .gitignore
```

---

## Key Results

### EUR/USD (Spot)
- Sharpe Ratio: ~0.24
- Return: ~2.9% (after ~20 years)
- Drawdown: Low, stable

### 6E Futures
- Sharpe Ratio: ~0.31
- Return: ~4.5%

### Options Spread Strategy (6E)
- Total Return: **-0.96%** (early prototype)
- Sharpe Ratio: **-7.76** (not yet profitable)
- Insight: Volatility overpricing may not be consistently exploitable via naive spread entries

### USD/JPY Extension (Work in Progress)
- Return: 7.57%
- Sharpe: 0.17
- Max Drawdown: 4.35%

---

## Tech Stack

- Python 3.12+
- Pandas / NumPy / Matplotlib / Seaborn
- Jupyter Notebooks for visual strategy diagnostics
- Git + GitHub (SSH linked)
- Interactive Brokers API (via `ib_insync`, to be integrated)

---

## Status & Milestones

- [x] Fetch 20 years of EUR/USD and 6E data
- [x] Build modular backtest engine
- [x] Design mean-reversion signal
- [x] Implement spread logic (BSM-based pricing)
- [x] Extend to USD/JPY (ongoing)
- [ ] Integrate IBKR API live trading (post backtest)
- [ ] Parameter optimization + Monte Carlo
- [ ] Deploy live signals via Telegram Bot

---


## Author
**Nicholas Hong**  
Built for educational and research purposes. Not financial advice.

## Acknowledgments
Special thanks to Jirong Huang, Quantitative Portfolio Manager at Eastspring Investments, for his guidance and mentorship throughout the development of this strategy. His insights into FX markets and options structuring were instrumental in shaping the project.
