# Futures Options Strategy Backtester

This project is a Python-based backtesting engine for a directional options strategy on forex futures.  
It is based on the volatility risk premium and mean-reversion approach outlined in a private strategy document.

---

## Strategy Logic (Summary)

- Use **regression slope** to determine trend direction
- Use **Bollinger Bands** to detect overbought/oversold zones
- Enter **bear put spread** at lower band on uptrend
- Enter **bull call spread** at upper band on downtrend
- Track cumulative P&L and simulate realistic spreads (coming soon)

---

## Folder Structure
<pre>
trading-bot/
├── README.md                # Project overview and documentation
├── backtest.py              # Main backtest runner
├── data/                    # CSVs and market data
├── strategies/              # Custom strategy logic (mean reversion, etc.)
└── utils/                   # Indicators, tools (regression, Bollinger, etc.)
</pre>

---

## Tech Stack

- Python (3.8+)
- Pandas / NumPy / Matplotlib
- yFinance (for quick data), IBKR / Databento later
- Git + GitHub (SSH enabled)

---

## Status

- [x] Project initialized
- [x] SSH + GitHub working
- [x] Dummy backtest engine running
- [ ] Load real futures data
- [ ] Implement strategy logic
- [ ] Backtest full pipeline
- [ ] Auto-trade integration (IBKR)

---

## Author

Nicholas Hong | Built for educational and research purposes. Not financial advice.


