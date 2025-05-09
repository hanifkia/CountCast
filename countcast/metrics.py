import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def forecast_accuracy(
    df,
    groupby="group_id",
    actual_col="value",
    forecast_col="forecast",
    lower_col="lower",
    upper_col="upper",
    confidence_level=0.9,
):
    """
    Calculate accuracy metrics for forecasts, grouped by a specified column.

    Args:
        df (pd.DataFrame): DataFrame with actual values, forecast values, group IDs, and bounds.
        groupby (str): Column name for grouping (e.g., 'group_id').
        actual_col (str): Column name for actual values.
        forecast_col (str): Column name for forecast values.
        lower_col (str): Column name for lower bound of confidence interval.
        upper_col (str): Column name for upper bound of confidence interval.
        confidence_level (float): Confidence level for coverage calculation (e.g., 0.9).

    Returns:
        dict: Dictionary with group IDs as keys and metrics (SMAPE, MAPE, Coverage) as values.
    """
    grouped = df.groupby(groupby)
    results = {}

    for group_id, group_df in grouped:
        group_df = group_df.dropna(subset=[actual_col, forecast_col])
        group_df[actual_col] = pd.to_numeric(group_df[actual_col], errors="coerce")
        group_df[forecast_col] = pd.to_numeric(group_df[forecast_col], errors="coerce")
        group_df = group_df.dropna(subset=[actual_col, forecast_col])

        actual = group_df[actual_col].values
        forecast = group_df[forecast_col].values

        if len(actual) < 2:
            results[group_id] = {"SMAPE": np.nan, "MAPE": np.nan, "Coverage": np.nan}
            continue

        mse = mean_squared_error(actual, forecast)
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)
        smape = (
            np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))
            * 100
        )
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100

        if lower_col in group_df.columns and upper_col in group_df.columns:
            group_df[lower_col] = pd.to_numeric(group_df[lower_col], errors="coerce")
            group_df[upper_col] = pd.to_numeric(group_df[upper_col], errors="coerce")
            group_df = group_df.dropna(subset=[lower_col, upper_col])
            lower = group_df[lower_col].values
            upper = group_df[upper_col].values
            coverage = np.mean((actual >= lower) & (actual <= upper))
        else:
            coverage = np.nan

        results[group_id] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "SMAPE": smape,
            "MAPE": mape,
            "Coverage": coverage,
        }

    return results
