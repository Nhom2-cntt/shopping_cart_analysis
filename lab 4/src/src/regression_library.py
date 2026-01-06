import pandas as pd # type: ignore
import numpy as np # type: ignore

from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore


class RegressionDatasetBuilder:
    """
    Prepare dataset for regression forecasting
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_target(self, target_col: str, horizon: int):
        self.df[f"{target_col}_target"] = self.df[target_col].shift(-horizon)
        return self

    def drop_na(self):
        self.df = self.df.dropna()
        return self.df


class TimeSeriesSplitter:
    """
    Split train/test by time cutoff
    """
    def __init__(self, df: pd.DataFrame, cutoff: str):
        self.df = df
        self.cutoff = cutoff

    def split(self, feature_cols: list, target_col: str):
        train = self.df[self.df["datetime"] < self.cutoff]
        test = self.df[self.df["datetime"] >= self.cutoff]

        X_train = train[feature_cols]
        y_train = train[target_col]

        X_test = test[feature_cols]
        y_test = test[target_col]

        return X_train, X_test, y_train, y_test


class RegressionBaseline:
    """
    Baseline regression model
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RegressionEvaluator:
    """
    Evaluate regression forecast
    """
    @staticmethod
    def evaluate(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        return {
            "MAE": mae,
            "RMSE": rmse
        }
