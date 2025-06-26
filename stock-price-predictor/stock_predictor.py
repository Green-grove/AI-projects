import yfinance as yf
import pandas as pd

# Fetch data for Apple (AAPL) as a pandas.DataFrame object from 2015 to 2025
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2025-06-24")

# Save data in CSV format on disc for later use under this file name "{ticker}_stock_data.csv"
data.to_csv(f"{ticker}_stock_data.csv")
print(data.head(10))