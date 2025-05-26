from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=3)

# Try exact known contract (e.g., 6E June 2024)
contract = Future(symbol='6E', lastTradeDateOrContractMonth='202406', exchange='GLOBEX', currency='USD')
details = ib.reqContractDetails(contract)

if details:
    print("✅ Found contract:")
    print(f"• {details[0].contract.localSymbol} — {details[0].contract.lastTradeDateOrContractMonth}")
else:
    print("❌ No contract details returned. Check permissions and contract month.")
