import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dummy data (for now, until we load real futures data)
dates = pd.date_range(start="2023-01-01", periods=100)
price = np.cumsum(np.random.randn(100)) + 100  # random walk
df = pd.DataFrame({"Date": dates, "Close": price}).set_index("Date")

# Simple strategy: Buy if yesterday was down, sell if it was up
df["Signal"] = np.where(df["Close"].diff() < 0, 1, -1)
df["Return"] = df["Close"].pct_change()
df["Strategy"] = df["Signal"].shift(1) * df["Return"]

# Cumulative returns
df["Equity"] = (1 + df["Strategy"]).cumprod()

# Plot
df[["Close", "Equity"]].plot(title="Price vs Strategy Equity Curve")
plt.show()