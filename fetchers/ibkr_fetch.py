from ib_insync import *
import pandas as pd
import os

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Spot EUR/USD contract
contract = Forex('EURUSD')

# Request 20 years of daily bars
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='20 Y',
    barSizeSetting='1 day',
    whatToShow='MIDPOINT',
    useRTH=False,
    formatDate=1
)

# Convert to DataFrame and save
df = util.df(bars)
os.makedirs("data", exist_ok=True)
df.to_csv("data/EURUSD_IBKR.csv", index=False)
print("✅ Downloaded 20Y EUR/USD spot data")

ib.disconnect()
