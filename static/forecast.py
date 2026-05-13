import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv("asda_daily_sales_with_moving_avg.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Prepare time series
time_series = df['prices_(£)']

# Exponential Smoothing
model = SimpleExpSmoothing(time_series, initialization_method='heuristic').fit(smoothing_level=0.2, optimized=False)
df['SES_Forecast'] = model.fittedvalues

# Forecast next 7 days
forecast = model.forecast(7)
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df['prices_(£)'], label='Actual Sales')
plt.plot(df.index, df['SES_Forecast'], label='SES Forecast', linestyle='--')
plt.plot(forecast_dates, forecast, label='7-Day Forecast', linestyle='dotted')
plt.xlabel('Date')
plt.ylabel('Sales (£)')
plt.title('Exponential Smoothing Forecast - ASDA Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
