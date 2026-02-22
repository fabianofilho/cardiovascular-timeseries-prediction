from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Forecaster(ABC):
    name: str

    @abstractmethod
    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        raise NotImplementedError


class SarimaForecaster(Forecaster):
    name = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.forecast(steps=horizon)
        return np.asarray(pred, dtype=float)


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self):
        from prophet import Prophet

        self._prophet_cls = Prophet

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        df = pd.DataFrame({"ds": train.index, "y": train.values})
        model = self._prophet_cls(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        model.fit(df)

        freq = pd.infer_freq(train.index)
        if freq is None:
            freq = "MS"

        future = model.make_future_dataframe(periods=horizon, freq=freq)
        fcst = model.predict(future).tail(horizon)
        return fcst["yhat"].to_numpy(dtype=float)


class TimesFMForecaster(Forecaster):
    name = "timesfm"

    def __init__(self):
        import timesfm

        self._timesfm = timesfm

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        # API do TimesFM pode variar por versão; tentamos as interfaces mais comuns.
        values = train.to_numpy(dtype=float)

        if hasattr(self._timesfm, "TimesFm"):
            model = self._timesfm.TimesFm(
                context_len=max(len(values), 32),
                horizon_len=horizon,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
            )
            if hasattr(model, "forecast"):
                preds, _ = model.forecast([values], freq=[0])
                return np.asarray(preds[0][:horizon], dtype=float)

        raise RuntimeError(
            "Interface TimesFM não reconhecida para esta versão. "
            "Ajuste o wrapper em src/cv_timeseries/models.py"
        )
