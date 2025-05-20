import yfinance as yf
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Symbol for EUR/USD Futures
symbol = "6E=F"

# Download daily data
df = yf.download(symbol, start="2018-01-01", end="2024-01-01", interval="1d")

# Save to CSV
df.to_csv("data/6E.csv")

print("✅ EUR/USD Futures saved to data/6E.csv")