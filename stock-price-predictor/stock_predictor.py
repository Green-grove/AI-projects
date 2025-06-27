import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# import, proces and store stock data and train linear regression model 

# Fetch data for Apple (AAPL) as a pandas.DataFrame object from 2015 to 2025
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2025-06-24")

# yf.download() returns a DataFrame with multi-index columns where each column label/name is a tuple, ex 
# [('Price', 'Close', 'AAPL'), ...]. This confused the convertion into CSV format. We need to flatten the tuple 
# lables before saving to CSV. This means the tubles into single string names like [('Price_Close_AAPL'), ...], 
# so CSV looks clean and easy to read.

# Flatten multi-index columns by joining names with underscore or space. data.columns.values emply we are looking 
# at the tupples that label each collumb and not the value in the cells that are accessed by data.values.
data.columns = ['_'.join(col).strip() for col in data.columns.values]

# Save flattened dataframe data in CSV format on disc for later use under this file name "{ticker}_stock_data.csv"
data.to_csv(f"{ticker}_stock_data.csv")

# Load CSV back properly, parse the index (Date) as datetime
data_loaded = pd.read_csv(
    f"{ticker}_stock_data.csv",
    index_col=0,
    parse_dates=True  # pandas can infer the format now
)

# Rename flattened pandas.DataFrame headers ex Close_APPL for future ease in data requests
rename_map = {
    f'Close_{ticker}': 'Close',
    f'Open_{ticker}': 'Open',
    f'High_{ticker}': 'High',
    f'Low_{ticker}': 'Low',
    f'Volume_{ticker}': 'Volume'
}
data.rename(columns=rename_map, inplace=True)

# Create a new collumb in pandas.DataFrame 7 and 30 day moving average  and drop first 6 and 29 days with NaN i.e Null value
data["MA7"] = data["Close"].rolling(window=7).mean()
data["MA30"] = data["Close"].rolling(window=30).mean()
data = data.dropna()

# Select features and taget for Linear regression model X are a matrix of input features and y. I had to use double brackets
# because features = data["Close","MA7","MA30"] will look for a nonexisting tuble collumb header ("Close","MA7","MA30"). double 
# brackets inply individual collumb headers. (This seems to include the date as first collumb that disapears in MinMaxScale) 
X = data[["Close","MA7","MA30"]] # features

# Shift all the target y values one step *up* so that each X[i] (features for day i) is paired with the target y[i] = Close 
# price on day i+1. dropna() removes the last row, which has no corresponding y value after shifting.
y = data["Close"].shift(-1).dropna()  
# To use this data on the model i need to scale the data to a MinMax value from 0 to 1. 
# ex If min = 10 is 0 and Max = 50 is 1 this results in MinMax[10,20,30,40,50] = [0, 0.25 , 0.5 , 0.75 , 1]  
X_scaled = MinMaxScaler().fit_transform(X)

# Remove last row from features to match length of y
X_scaled = X_scaled[:-1]

# Save data with assigned index labels from y (index=y.index) to the new DataFrame of feature. 
# to keep the rows alignment consistent between your features X and your target y.
pd.DataFrame(X_scaled, columns=["Close","MA7","MA30"], index=y.index).to_csv("preprocessed_data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("\n")
print("First 5 lines of data:") # print 5 lines of the code
print(data_loaded.head())
print("\n")
print("Number of null cells in data:") # Check for missing values
print(data_loaded.isnull().sum())
print("\n")
print("X")
print(X)
print("X.scaled")
print(X_scaled)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.savefig("prediction_plot.png")
plt.show()

