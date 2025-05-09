import pandas as pd
from pandas.tseries.offsets import MonthEnd


def add_conditional_columns(df, month_end_start_days):
    """
    Add boolean columns for conditional seasonalities.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column.
        month_end_start_days (int): Days to consider as month end/start.

    Returns:
        pd.DataFrame: DataFrame with added conditional columns.
    """

    def is_weekend(ds):
        date = pd.to_datetime(ds)
        return date.weekday() >= 5  # Saturday (5) or Sunday (6)

    def is_month_end_start(ds):
        date = pd.to_datetime(ds)
        day = date.day
        days_in_month = (date + MonthEnd(0)).day
        return (day <= month_end_start_days) or (
            day > days_in_month - month_end_start_days
        )

    df = df.copy()
    df["is_weekend"] = df["timestamp"].apply(is_weekend)
    df["is_weekday"] = ~df["is_weekend"]
    df["is_month_end_start"] = df["timestamp"].apply(is_month_end_start)
    df["is_month_rest"] = ~df["is_month_end_start"]
    return df
