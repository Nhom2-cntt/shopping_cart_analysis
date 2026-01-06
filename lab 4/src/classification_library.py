import zipfile
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss


class AirQualityDataLoader:
    """
    Load Beijing air quality data from ZIP file
    """
    def __init__(self, zip_path: str):
        self.zip_path = zip_path

    def load_from_zip(self) -> pd.DataFrame:
        dfs = []

        with zipfile.ZipFile(self.zip_path, "r") as z:
            for file in z.namelist():
                if file.endswith(".csv"):
                    df = pd.read_csv(z.open(file))
                    dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        return data


class AirQualityCleaner:
    """
    Clean and preprocess air quality data
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_datetime(self):
        self.df["datetime"] = pd.to_datetime(
            self.df[["year", "month", "day", "hour"]]
        )
        return self

    def clean_pm25(self):
        # 999 = invalid value
        self.df["pm2.5"] = self.df["pm2.5"].replace(999, np.nan)
        self.df["pm2.5"] = self.df["pm2.5"].fillna(method="ffill")
        return self

    def add_time_features(self):
        self.df["hour_of_day"] = self.df["datetime"].dt.hour
        self.df["day_of_week"] = self.df["datetime"].dt.dayofweek
        self.df["month"] = self.df["datetime"].dt.month
        return self

    def finalize(self):
        self.df = self.df.sort_values("datetime")
        return self.df


class TimeSeriesDiagnostics:
    """
    Stationarity & missing diagnostics
    """
    @staticmethod
    def missing_ratio(df: pd.DataFrame) -> pd.Series:
        return df.isna().mean().sort_values(ascending=False)

    @staticmethod
    def adf_test(series: pd.Series) -> dict:
        result = adfuller(series.dropna())
        return {
            "adf_stat": result[0],
            "p_value": result[1]
        }

    @staticmethod
    def kpss_test(series: pd.Series) -> dict:
        stat, p_value, _, _ = kpss(series.dropna(), regression="c")
        return {
            "kpss_stat": stat,
            "p_value": p_value
        }


class FeatureEngineer:
    """
    Create lag features for regression baseline
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_lags(self, col: str, lags: list):
        for lag in lags:
            self.df[f"{col}_lag_{lag}"] = self.df[col].shift(lag)
        return self.df


def save_processed_data(df: pd.DataFrame, path: str):
    df.to_parquet(path)
