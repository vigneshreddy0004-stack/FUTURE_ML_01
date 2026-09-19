"""
FUTURE_ML_01 - Sales & Demand Forecasting
Run: python sales_forecasting.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_FILE = "sales_data.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["Date"]).sort_values("Date")
df = df.dropna().copy()

# Time-based feature engineering
df["day_of_week"] = df["Date"].dt.dayofweek
df["day_of_month"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["quarter"] = df["Date"].dt.quarter
df["year"] = df["Date"].dt.year
df["trend"] = np.arange(len(df))

# Lag and rolling features
df["lag_1"] = df["Sales"].shift(1)
df["lag_7"] = df["Sales"].shift(7)
df["lag_14"] = df["Sales"].shift(14)
df["rolling_7"] = df["Sales"].shift(1).rolling(7).mean()
df["rolling_30"] = df["Sales"].shift(1).rolling(30).mean()
df = df.dropna().reset_index(drop=True)

features = [
    "day_of_week", "day_of_month", "month", "quarter", "year",
    "trend", "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_30"
]

# Time-aware split: first 80% for training, final 20% for testing
split = int(len(df) * 0.80)
train, test = df.iloc[:split], df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)
model.fit(train[features], train["Sales"])
pred = model.predict(test[features])

mae = mean_absolute_error(test["Sales"], pred)
rmse = np.sqrt(mean_squared_error(test["Sales"], pred))
mape = np.mean(np.abs((test["Sales"] - pred) / test["Sales"])) * 100

print("Sales Forecasting Results")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

# Business-friendly forecast chart
plt.figure(figsize=(12, 5))
plt.plot(train["Date"].tail(120), train["Sales"].tail(120), label="Historical Sales")
plt.plot(test["Date"], test["Sales"], label="Actual Test Sales")
plt.plot(test["Date"], pred, label="Forecast")
plt.title("Sales Forecast - Historical vs Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_result.png", dpi=160)
plt.show()

# Feature importance
importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nTop drivers:")
print(importance.head(5))

plt.figure(figsize=(9, 5))
importance.head(8).sort_values().plot(kind="barh")
plt.title("Top Forecast Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=160)
plt.show()

# Simple business insights
avg_sales = df["Sales"].mean()
peak_month = df.groupby("month")["Sales"].mean().idxmax()
print("\nBusiness Insights")
print(f"Average daily sales: {avg_sales:.2f}")
print(f"Highest average-sales month number: {peak_month}")
print("Use the forecast to support inventory, staffing, and promotion planning.")
