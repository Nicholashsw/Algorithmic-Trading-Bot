# Volatility Risk Premium (VRP) Multi-Market Strategy  

A systematic FX & commodity options strategy designed to combine mean-reversion and trend-following signals with volatility risk premium harvesting, tested across spot FX, currency futures, and gold futures.  

---

## Strategy Overview  

This project explores how price signals can be structured into options spread strategies across:  

- **Spot FX** – EUR/USD, USD/JPY, GBP/USD, AUD/USD  
- **Futures** – 6E (Euro FX), 6J (Japanese Yen), USD/CHF  
- **Gold Futures** – GC as a safe-haven VRP candidate  

### Strategy Types  
- **Mean-Reversion** (Slope + Bollinger Bands) → Bull Call / Bear Put Spreads  
- **Trend-Following** (200-day MA) → Bull Put / Bear Call Spreads (20–10 delta)  
- **Iron Condor Variant** → Selling both spreads for rangebound conditions  

### Strategy Variants  

| Variant              | Logic                                  | Instrument Scope   | Spread Structure              | Delta Target |
|----------------------|----------------------------------------|--------------------|--------------------------------|--------------|
| VRP Mean Reversion   | Slope + BB for overbought/oversold     | Spot & Futures FX  | Bull Call / Bear Put           | N/A          |
| VRP Trend Following  | 200MA filter for trend bias            | FX & Gold Options  | Bull Put / Bear Call           | 20 / 10      |
| VRP Iron Condor      | Sell both spreads in rangebound        | FX & Gold Options  | Short Call + Short Put Spreads | 20 / 10      |  

---

## Data Sources  

- **Spot FX** → IBKR API (`ib_insync`)  
- **Futures Prices** → Bloomberg Terminal, CME data  
- **Options Chains** → Bloomberg (strike, bid/ask, IV, DTE, risk-free rate), Databento, CBOE  
- **Greeks** → Calculated via Black-76 / Black-Scholes  

---

## Pipeline  

1. Fetch historical & live data from IBKR, Bloomberg, Databento, CBOE  
2. Generate trading signals using mean-reversion & trend-following filters  
3. Price spreads via Black-76 for futures options  
4. Backtest with performance metrics: P&L, Sharpe, drawdown  
5. Apply risk filters: volatility, macro events  
6. Execute live or paper trades via IBKR API  

**Live Implementation Plan:**  
- Event-driven execution via `ib_insync` for continuous market data subscription and order placement  
- Cron jobs for batch logging (option chains, EOD summaries)  
- Safety controls: daily loss limits, per-symbol caps, auto-kill on disconnect  

---

## Results (Current Stage)  

| Asset / Strategy     | Return  | Sharpe | Max DD |
|----------------------|---------|--------|--------|
| EUR/USD Spot (20y)   | 2.9%    | 0.24   | Low    |
| 6E Futures           | 4.5%    | 0.31   | Low    |
| USD/JPY Spot         | 7.57%   | 0.17   | 4.35%  |
| 6E Options Prototype | -0.96%  | -7.76  | High   |  

*Note: Options stage is in prototype; entry filters & delta targeting are still in progress.*  

---

## Planned Research Extensions  

- Complete options chain ingestion for CHF & Gold  
- Add macro event-driven entry filters (FOMC, CPI, NFP)  
- Extend VRP strategies to Iron Condor framework  
- Integrate volatility and IV-rank filters  
- Test VRP strategies in equity index options (SPX, SPY)  
- Automate paper/live trades via IBKR API  
- Publish visual dashboards and Medium updates  

---

## Tech Stack  

Python 3.12 · Pandas · NumPy · Matplotlib · Seaborn · Jupyter · IBKR API · Bloomberg Terminal · Databento · CBOE  

---

## Acknowledgments  

Special thanks to **Huang Jirong**, Quantitative Portfolio Manager at Eastspring Investments, for mentorship and valuable insights into FX markets and options structuring.  

---

## Connect & Explore  

**Repository:** [github.com/Nicholashsw/trading-bot](https://github.com/Nicholashsw/trading-bot)  
**Portfolio:** [nicholashong.dev](https://nicholashong.dev)  
**Medium:** [medium.com/@nhong001](https://medium.com/@nhong001)  
**LinkedIn:** [sg.linkedin.com/in/nicholas-hong001](https://sg.linkedin.com/in/nicholas-hong001)  
**Email:** nhong001@e.ntu.edu.sg  

---
