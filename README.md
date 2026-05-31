# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 30-05-26

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

%matplotlib inline

plt.style.use('default')

data = pd.read_csv(
    "C:/Users/admin/OneDrive/Desktop/index_1.csv"
)

data['datetime'] = pd.to_datetime(
    data['date'] + ' ' + data['datetime'],
    errors='coerce'
)

data = data.dropna(subset=['datetime'])

data = data.sort_values('datetime')

data.set_index('datetime', inplace=True)

data['money'] = pd.to_numeric(
    data['money'],
    errors='coerce'
)

data = data.dropna(subset=['money'])

data_monthly = data['money'].resample('MS').mean()

data_monthly = data_monthly.fillna(method='ffill')

def arima_model(data, order):

    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]

    test_data = data[train_size:]

    model = ARIMA(
        train_data,
        order=order
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(
        steps=len(test_data)
    )

    rmse = np.sqrt(
        mean_squared_error(
            test_data,
            forecast
        )
    )

    mean_value = data.mean()

    std_value = data.std()

    plt.figure(figsize=(12, 6))

    plt.plot(
        train_data.index,
        train_data,
        label='Training Data'
    )

    plt.plot(
        test_data.index,
        test_data,
        label='Testing Data'
    )

    plt.plot(
        test_data.index,
        forecast,
        label='Forecasted Data'
    )

    plt.xlabel('Date')

    plt.ylabel('Coffee Sales')

    plt.title(
        'ARIMA Forecasting for Coffee Sales'
    )

    plt.legend()

    plt.grid(True)

    plt.show()

    print(
        "Root Mean Squared Error (RMSE):",
        rmse
    )

    print(
        "Mean Value:",
        mean_value
    )

    print(
        "Standard Deviation:",
        std_value
    )

arima_model(
    data_monthly,
    order=(1,1,1)
)
```
### OUTPUT:

<img width="983" height="594" alt="image" src="https://github.com/user-attachments/assets/beb6438c-9639-4a57-8023-03c4dd347c59" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
