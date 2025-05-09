from datetime import timedelta

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import MSTL

from .utils import add_conditional_columns


class CountForecaster:
    """A class to forecast integer counts in stationary time series using MSTL, Prophet, and SARIMA."""

    def __init__(
        self,
        prediction_length,
        seasonal_period=7,
        weekly_fourier_order=3,
        monthly_fourier_order=5,
        weekly_prior_scale=10.0,
        monthly_prior_scale=10.0,
        month_end_start_days=3,
    ):
        """
        Initialize the forecaster.

        Args:
            prediction_length (int): Number of days to forecast.
            seasonal_period (int): Seasonal period for SARIMA (e.g., 7 for weekly, 30 for monthly).
            weekly_fourier_order (int): Fourier order for weekly seasonalities.
            monthly_fourier_order (int): Fourier order for monthly seasonalities.
            weekly_prior_scale (float): Prior scale for weekly seasonalities.
            monthly_prior_scale (float): Prior scale for monthly seasonalities.
            month_end_start_days (int): Days to consider as month end/start (e.g., last 3 and first 3).
        """
        self.prediction_length = prediction_length
        self.seasonal_period = seasonal_period
        self.weekly_fourier_order = weekly_fourier_order
        self.monthly_fourier_order = monthly_fourier_order
        self.weekly_prior_scale = weekly_prior_scale
        self.monthly_prior_scale = monthly_prior_scale
        self.month_end_start_days = month_end_start_days
        self.prophet_trend_model = None
        self.prophet_seasonal_model = None
        self.sarima_model = None
        self.data = None
        self.mstl = None
        self.start_time = None

    def fit(self, data):
        """
        Fit the forecasting model to the data.

        Args:
            data (pd.DataFrame): DataFrame with 'timestamp' (datetime) and 'value' (integer counts).

        Raises:
            ValueError: If data is invalid or contains gaps.
        """
        self.data = data.copy()
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])

        # Validate data
        expected_dates = pd.date_range(
            start=self.data["timestamp"].min(),
            end=self.data["timestamp"].max(),
            freq="D",
        )
        if len(self.data) != len(expected_dates):
            raise ValueError(
                "Data contains gaps or non-consecutive timestamps. Ensure daily frequency."
            )
        if not np.issubdtype(self.data["value"].dtype, np.integer):
            raise ValueError("Value column must contain integer counts.")

        self.start_time = self.data["timestamp"].max() + timedelta(days=1)

        # Add conditional columns
        self.data = add_conditional_columns(self.data, self.month_end_start_days)

        # MSTL decomposition
        self.mstl = MSTL(self.data["value"], periods=[7, 30]).fit()
        self.data["trend"] = self.mstl.trend
        self.data["residual"] = self.mstl.resid
        self.data["no_trend"] = self.data["value"] - self.data["trend"]

        # Prophet: Trend forecast
        trend_df = self.data[["timestamp", "trend"]].rename(
            columns={"timestamp": "ds", "trend": "y"}
        )
        self.prophet_trend_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            interval_width=0.99,
        )
        self.prophet_trend_model.fit(trend_df)

        # Prophet: Conditional seasonalities
        seasonal_df = self.data[
            [
                "timestamp",
                "no_trend",
                "is_weekend",
                "is_weekday",
                "is_month_end_start",
                "is_month_rest",
            ]
        ].copy()
        seasonal_df = seasonal_df.rename(columns={"timestamp": "ds", "no_trend": "y"})
        self.prophet_seasonal_model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            interval_width=0.99,
        )
        self.prophet_seasonal_model.add_seasonality(
            name="weekly_weekend",
            period=7,
            fourier_order=self.weekly_fourier_order,
            condition_name="is_weekend",
            prior_scale=self.weekly_prior_scale,
        )
        self.prophet_seasonal_model.add_seasonality(
            name="weekly_weekday",
            period=7,
            fourier_order=self.weekly_fourier_order,
            condition_name="is_weekday",
            prior_scale=self.weekly_prior_scale,
        )
        self.prophet_seasonal_model.add_seasonality(
            name="monthly_end_start",
            period=30.42,
            fourier_order=self.monthly_fourier_order,
            condition_name="is_month_end_start",
            prior_scale=self.monthly_prior_scale,
        )
        self.prophet_seasonal_model.add_seasonality(
            name="monthly_rest",
            period=30.42,
            fourier_order=self.monthly_fourier_order,
            condition_name="is_month_rest",
            prior_scale=self.monthly_prior_scale,
        )
        self.prophet_seasonal_model.fit(seasonal_df)

        # SARIMA: Residual forecast
        try:
            sarima_model = auto_arima(
                self.data["residual"],
                seasonal=True,
                m=self.seasonal_period,
                max_D=1,
                max_d=1,
                trace=False,
                suppress_warnings=True,
                error_action="warn",
            )
            best_order = sarima_model.order
            best_seasonal_order = sarima_model.seasonal_order
            self.sarima_model = ARIMA(
                self.data["residual"],
                order=best_order,
                seasonal_order=best_seasonal_order,
            ).fit()
        except ValueError as e:
            print(f"SARIMA fitting failed: {e}. Falling back to non-seasonal ARIMA.")
            sarima_model = auto_arima(
                self.data["residual"],
                seasonal=False,
                max_d=1,
                trace=False,
                suppress_warnings=True,
            )
            best_order = sarima_model.order
            self.sarima_model = ARIMA(self.data["residual"], order=best_order).fit()

    def get_future_dates(self):
        """Generate future dates for forecasting."""
        end = self.start_time + timedelta(days=(self.prediction_length - 1))
        return pd.DataFrame(
            {"timestamp": pd.date_range(start=self.start_time, end=end, freq="D")}
        )

    def predict(self):
        """
        Generate forecasts for future dates.

        Returns:
            pd.DataFrame: Forecast with 'timestamp', 'lower', 'forecast', 'upper'.
        """
        future = self.get_future_dates()
        future["timestamp"] = pd.to_datetime(future["timestamp"])
        future = add_conditional_columns(future, self.month_end_start_days)

        # Prophet: Trend forecast
        future_trend = future[["timestamp"]].rename(columns={"timestamp": "ds"})
        trend_forecast = self.prophet_trend_model.predict(future_trend)
        trend_forecast = trend_forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].rename(
            columns={
                "ds": "timestamp",
                "yhat": "trend_forecast",
                "yhat_lower": "trend_lower",
                "yhat_upper": "trend_upper",
            }
        )

        # Prophet: Conditional seasonalities forecast
        future_seasonal = future[
            [
                "timestamp",
                "is_weekend",
                "is_weekday",
                "is_month_end_start",
                "is_month_rest",
            ]
        ].rename(columns={"timestamp": "ds"})
        seasonal_forecast = self.prophet_seasonal_model.predict(future_seasonal)
        seasonal_forecast = seasonal_forecast[["ds", "yhat"]].rename(
            columns={"ds": "timestamp", "yhat": "seasonal_forecast"}
        )

        # SARIMA: Residual forecast
        residual_forecast = self.sarima_model.forecast(steps=len(future))

        # Combine forecasts
        forecast = trend_forecast.merge(seasonal_forecast, on="timestamp")
        forecast["residual_forecast"] = residual_forecast.values
        forecast["forecast"] = (
            forecast["trend_forecast"].values
            + forecast["seasonal_forecast"].values
            + forecast["residual_forecast"].values
        )
        forecast["upper"] = (
            forecast["trend_upper"].values
            + forecast["seasonal_forecast"].values
            + forecast["residual_forecast"].max()
        )
        forecast["lower"] = (
            forecast["trend_lower"].values
            + forecast["seasonal_forecast"].values
            + forecast["residual_forecast"].min()
        )

        # Clip bounds to prevent extreme values
        eps = 0.1
        forecast["upper"] = forecast["upper"].clip(
            lower=forecast["forecast"] + eps * forecast["forecast"]
        )
        forecast["lower"] = forecast["lower"].clip(
            upper=forecast["forecast"] - eps * forecast["forecast"]
        )

        # Round to integers
        forecast["forecast"] = np.round(forecast["forecast"]).astype(int)
        forecast["upper"] = np.ceil(forecast["upper"]).astype(int)
        forecast["lower"] = np.floor(forecast["lower"]).astype(int)

        # Ensure non-negative forecasts
        forecast["forecast"] = np.maximum(0, forecast["forecast"])
        forecast["lower"] = np.maximum(0, forecast["lower"])
        forecast["upper"] = np.maximum(0, forecast["upper"])

        return forecast[["timestamp", "lower", "forecast", "upper"]]

    def plot_components(self):
        """Plot Prophet trend and seasonal components."""
        import matplotlib.pyplot as plt

        future = self.prophet_trend_model.make_future_dataframe(
            periods=self.prediction_length, freq="D"
        )
        trend_forecast = self.prophet_trend_model.plot_components(
            self.prophet_trend_model.predict(future)
        )

        future_seasonal = add_conditional_columns(
            pd.DataFrame({"timestamp": future["ds"]}), self.month_end_start_days
        )
        future_seasonal = future_seasonal[
            [
                "timestamp",
                "is_weekend",
                "is_weekday",
                "is_month_end_start",
                "is_month_rest",
            ]
        ].rename(columns={"timestamp": "ds"})
        seasonal_forecast = self.prophet_seasonal_model.plot_components(
            self.prophet_seasonal_model.predict(future_seasonal)
        )
        return trend_forecast, seasonal_forecast
