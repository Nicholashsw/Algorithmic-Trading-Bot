from ib_insync import *
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# EUR/USD spot FX contract
contract = Forex('EURUSD')

# Request 20 years of daily bars
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='20 Y',          # ✅ Up to 20 years
    barSizeSetting='1 day',
    whatToShow='MIDPOINT',       # Use 'MIDPOINT' or 'TRADES'
