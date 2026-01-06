import itertools
import numpy as np # type: ignore
import pandas as pd # type: ignore

from statsmodels.tsa.stattools import adfuller, kpss # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore

from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore


# =========================
# 1. STATIONARITY CHECK
# =========================
class StationarityTester:

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


# =========================
# 2. ACF / PACF PLOTTING
# =========================
class ACFPACFAnalyzer:

    @staticmethod
    def plot(series: pd.Series, lags: int = 48):
        plot_acf(series.dropna(), lags=lags)
        plot_pacf(series.dropna(), lags=lags, method="ywm")


# =========================
# 3. ARIMA GRID SEARCH
# =========================
class ARIMAGridSearch:

    def __init__(self, series: pd.Series, d: int):
        self.series = series
        self.d = d

    def search(self, p_range: range, q_range: range):
        results = []

        for p, q in itertools.product(p_range, q_range):
            try:
                model = ARIMA(self.series, order=(p, self.d, q))
                fitted = model.fit()

                results.append({
                    "p": p,
                    "d": self.d,
                    "q": q,
                    "aic": fitted.aic
                })

            except Exception:
                continue

        return pd.DataFrame(results).sort_values("aic")


# =========================
# 4. ARIMA FORECASTER
# =========================
class ARIMAForecaster:

    def __init__(self, order: tuple):
        self.order = order
        self.model = None

    def fit(self, series: pd.Series):
        self.model = ARIMA(series, order=self.order).fit()
        return self

    def forecast(self, steps: int):
        return self.model.forecast(steps=steps)


# =========================
# 5. EVALUATION
# ====
