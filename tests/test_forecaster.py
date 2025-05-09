import numpy as np
import pandas as pd
import pytest

from countcast.forecaster import CountForecaster


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2022-01-01", end="2022-12-31", freq="D"),
            "value": np.random.randint(10, 50, size=365),
        }
    )


def test_count_forecaster_fit_predict(sample_data):
    forecaster = CountForecaster(prediction_length=30, seasonal_period=7)
    forecaster.fit(sample_data)
    forecast = forecaster.predict()

    assert isinstance(forecast, pd.DataFrame)
    assert set(forecast.columns) == {"timestamp", "lower", "forecast", "upper"}
    assert len(forecast) == 30
    assert np.issubdtype(forecast["forecast"].dtype, np.integer)
    assert (forecast["forecast"] >= 0).all()
    assert (forecast["lower"] <= forecast["forecast"]).all()
    assert (forecast["upper"] >= forecast["forecast"]).all()
