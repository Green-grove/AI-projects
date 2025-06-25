import yfinance as yf
import pandas as pd

# Fetch data for Apple (AAPL) from 2015 to 2025
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2025-06-24")

# Save to CSV
data.to_csv(f"{ticker}_stock_data.csv")
print(data.head())
