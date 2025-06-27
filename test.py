import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample data: Close prices and some moving averages
data = pd.DataFrame({
    "Close": [100, 102, 104, 106, 108],
    "MA7": [None, None, 102, 104, 106],
    "MA30": [None, None, None, None, None]  # Just example, all NaNs here
})

# Select feature columns
features = ["Close", "MA7", "MA30"]
X = data[features]

# Shift target y by -1 (next day's close)
y = data["Close"].shift(-1)

print("Original X:")
print(X)

print("\nOriginal y (after shift):")
print(y)

# Drop NaN values in y (last one after shift)
y_clean = y.dropna()

print("\nCleaned y (after dropna):")
print(y_clean)

# Scale features X before alignment (scaling keeps all rows)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nX_scaled before alignment:")
print(X_scaled)

# Align X_scaled by removing last row to match y length
X_scaled_aligned = X_scaled[:-1]

print("\nX_scaled after alignment (last row removed):")
print(X_scaled_aligned)

# Also show original X aligned (for comparison)
X_aligned = X[:-1]

print("\nOriginal X after alignment (last row removed):")
print(X_aligned)

# Confirm lengths match
print(f"\nLength of X_scaled_aligned: {len(X_scaled_aligned)}")
print(f"Length of y_clean: {len(y_clean)}")
