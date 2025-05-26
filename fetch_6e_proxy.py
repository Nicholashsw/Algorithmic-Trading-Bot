import yfinance as yf

# Proxy for 6E continuous futures using EURUSD spot
df = yf.download("EURUSD=X", start="2004-01-01", end="2024-01-01", interval="1d")

if df.empty:
    print("❌ No data fetched.")
else:
    df.to_csv("data/6E_continuous_proxy.csv")
    print("✅ Saved as data/6E_continuous_proxy.csv")

