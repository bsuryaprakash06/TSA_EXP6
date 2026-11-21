# Ex.No: 6 HOLT WINTERS METHOD

## Date: 21 - 11 - 2026

# AIM
To implement Holt-Winters Exponential Smoothing for time series analysis, evaluate forecasting accuracy, and predict future temperatures using historical data.

# ALGORITHM

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions


# PROGRAM

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load the dataset, set daily frequency, and interpolate missing values
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

data = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
data = data.asfreq('D').interpolate(method='linear')

data.head()

# Step 2: Explore and plot data
X = data['Temp']

# Step 3: Check for seasonality
decomposition = seasonal_decompose(X, model='additive', period=365)

decomposition.plot()
plt.show()

# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(X.values.reshape(-1, 1)).flatten(),
    index=X.index
)

# Plot scaled data
scaled_data.plot()
plt.title('Scaled Daily Temperature Data')
plt.xlabel('Date')
plt.ylabel('Scaled Temperature')
plt.show()

# Step 4: Split train and test data
split = int(len(X) * 0.75)

train_raw = X[:split]
test_raw = X[split:]

# Step 5: Scale the data
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_raw.values.reshape(-1, 1)
).flatten() + 1

test_scaled = scaler.transform(
    test_raw.values.reshape(-1, 1)
).flatten() + 1

train_data = pd.Series(train_scaled, index=train_raw.index)
test_data = pd.Series(test_scaled, index=test_raw.index)

# Step 6: Create and train Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=365
).fit()

# Step 7: Predict test data
test_predictions_add = model_add.forecast(steps=len(test_data))

# Convert predictions back to original temperature scale
predictions = scaler.inverse_transform(
    (test_predictions_add.values - 1).reshape(-1, 1)
).flatten()

actual = scaler.inverse_transform(
    (test_data.values - 1).reshape(-1, 1)
).flatten()

# Step 8: Visual evaluation
plt.plot(train_raw, label='Train Data')
plt.plot(test_raw, label='Test Data')
plt.plot(test_raw.index, predictions, label='Predictions')

plt.title('Visual Evaluation')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Step 9: Evaluate the model
mae = mean_absolute_error(actual, predictions)
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

# Step 10: Create the final model using complete data
final_scaler = MinMaxScaler()

scaled_data = final_scaler.fit_transform(
    X.values.reshape(-1, 1)
).flatten() + 1

scaled_series = pd.Series(scaled_data, index=X.index)

final_model = ExponentialSmoothing(
    scaled_series,
    trend='add',
    seasonal='mul',
    seasonal_periods=365
).fit()

# Step 11: Predict future data for one year
final_predictions = final_model.forecast(steps=365)

future_values = final_scaler.inverse_transform(
    (final_predictions.values - 1).reshape(-1, 1)
).flatten()

future_dates = pd.date_range(
    start=X.index[-1] + pd.Timedelta(days=1),
    periods=365,
    freq='D'
)

future_series = pd.Series(future_values, index=future_dates)

# Step 12: Plot future predictions
plt.figure(figsize=(12, 6))

plt.plot(X, label='Original Data')
plt.plot(future_series, label='Future Predictions')

plt.title('Holt-Winters Future Temperature Forecast')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()
```

# OUTPUT

## Scaled Data Plot
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ed7c0783-5f64-48a8-a312-66e730f2d298" />

## Decomposed Plot
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/1a3c963e-817e-428f-96b0-e650ceffef0a" />

## Test Prediction
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/67755a87-0969-4566-982d-c0152dba15cc" />

## Model Performance Metrics
<img width="330" height="94" alt="{4D9726BC-56D4-4ED6-8D41-9674FEE3A0C2}" src="https://github.com/user-attachments/assets/9f2c4444-b00d-4518-84f3-90ddca6ac1c2" />


## Final Prediction
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/0f6d8208-d604-4321-9cd8-1adc8f827d66" />


# RESULT
Thus the program run successfully based on the Holt Winters Method model.
