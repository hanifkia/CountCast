# Countcast

`countcast` is a Python package for forecasting integer counts in stationary time series, such as order counts, event occurrences, or route demands. It combines MSTL decomposition, Prophet for trend and conditional seasonalities, and SARIMA for residuals, with support for weekly (weekend vs. weekday) and monthly (month-end/start vs. rest) seasonal patterns. The package ensures integer outputs and includes evaluation metrics like SMAPE, MAPE, and coverage.

## Features

- Forecast integer counts with conditional seasonalities.
- Modular pipeline using MSTL, Prophet, and SARIMA.
- Evaluation metrics for grouped forecasts (e.g., by region or category).
- Supports stationary time series with daily frequency.
- Easy-to-use API with robust error handling.

## Installation

```bash
pip install .
```

## Usage

```python
import pandas as pd
from countcast.forecaster import CountForecaster
from countcast.metrics import forecast_accuracy

# Load data
data = pd.read_csv("countcast/data/sample_data.csv")

# Initialize and fit forecaster
forecaster = CountForecaster(
    prediction_length=30,
    seasonal_period=7,
    weekly_fourier_order=3,
    monthly_fourier_order=5,
    weekly_prior_scale=10.0,
    monthly_prior_scale=10.0,
    month_end_start_days=3
)
forecaster.fit(data)

# Generate forecast
forecast = forecaster.predict()
print(forecast)

# Evaluate forecast (example with group_id)
forecast['group_id'] = 1
forecast['value'] = np.random.randint(10, 50, size=len(forecast))  # Simulated actuals
metrics = forecast_accuracy(forecast, groupby='group_id')
print(metrics)
```

## Directory Structure

```
countcast/
├── countcast/
│   ├── __init__.py
│   ├── forecaster.py
│   ├── metrics.py
│   ├── utils.py
│   └── data/
│       └── sample_data.csv
├── tests/
│   ├── __init__.py
│   └── test_forecaster.py
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies.

## Development

To run tests:

```bash
pytest tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please submit a pull request or open an issue on GitHub.

## Contact

For questions, contact [kia.hanif@gmail.com].
