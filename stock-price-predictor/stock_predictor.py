import time
import yfinance as yf
import pandas as pd

ticker = "AAPL"
retries = 3

for i in range(retries):
    try:
        data = yf.download(ticker, start="2015-01-01", end="2025-06-24", interval="1d", threads=False)
        if data.empty:
            raise ValueError("Download returned empty data.")
        break
    except Exception as e:
        print(f"Attempt {i+1} failed: {e}")
        time.sleep(10)  # wait 10 seconds before retrying
else:
    print("Download failed after multiple attempts.")
    data = pd.DataFrame()  # so rest of script doesn't break

print(data.head())
