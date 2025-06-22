from ib_insync import *
import pandas as pd
import os
import time
from datetime import datetime

# Setup
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=10)
os.makedirs("data", exist_ok=True)

# Generate contract months: March, June, Sep, Dec each year
years = list(range(2004, datetime.now().year + 1))
months = ['03', '06', '09', '12']
contracts = [f"{y}{m}" for y in years for m in months]

all_dfs = []

for expiry in contracts:
    try:
        contract = Future(symbol='6E', lastTradeDateOrContractMonth=expiry, exchange='GLOBEX', currency='USD')
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        df = util.df(bars)
        if not df.empty:
            df["contract"] = expiry
            all_dfs.append(df)
            print(f"✅ {expiry}: {len(df)} rows")
        else:
            print(f"⚠️ {expiry}: No data")
        time.sleep(2.5)  # IBKR pacing rules
    except Exception as e:
        print(f"❌ {expiry} error: {e}")
        continue

# Combine all
if all_dfs:
    combined_df = pd.concat(all_dfs)
    combined_df.set_index('date', inplace=True)
    combined_df.sort_index(inplace=True)
    combined_df.to_csv("data/6E_continuous.csv")
    print("✅ Saved 6E continuous futures data to data/6E_continuous.csv")
else:
    print("❌ No data was collected.")

ib.disconnect()
