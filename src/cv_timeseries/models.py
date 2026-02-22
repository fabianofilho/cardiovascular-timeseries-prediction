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
        self._model = None
        self._model_horizon = None
        self._repo_used = None

    def _build_model(self, horizon: int):
        repo_candidates = [
            "google/timesfm-2.0-500m-pytorch",
            "google/timesfm-1.0-200m-pytorch",
        ]

        if hasattr(self._timesfm, "TimesFmHparams") and hasattr(
            self._timesfm, "TimesFmCheckpoint"
        ):
            last_error = None
            for repo_id in repo_candidates:
                try:
                    hparams = self._timesfm.TimesFmHparams(
                        backend="cpu",
                        context_len=512,
                        horizon_len=horizon,
                        per_core_batch_size=1,
                    )
                    checkpoint = self._timesfm.TimesFmCheckpoint(
                        huggingface_repo_id=repo_id
                    )
                    model = self._timesfm.TimesFm(
                        hparams=hparams,
                        checkpoint=checkpoint,
                    )
                    self._repo_used = repo_id
                    return model
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(
                "Falha ao carregar checkpoint TimesFM via Hugging Face: "
                f"{last_error}"
            )

        raise RuntimeError("API TimesFMHparams/TimesFMCheckpoint não encontrada.")

    def forecast(self, train: pd.Series, horizon: int) -> np.ndarray:
        values = train.to_numpy(dtype=float)

        # Recarrega se o horizonte mudar entre execuções.
        if self._model is None or self._model_horizon != horizon:
            self._model = self._build_model(horizon=horizon)
            self._model_horizon = horizon

        if hasattr(self._model, "forecast"):
            preds, _ = self._model.forecast([values], freq=[0])
            return np.asarray(preds[0][:horizon], dtype=float)

        raise RuntimeError(
            "Interface TimesFM não reconhecida para esta versão. "
            "Ajuste o wrapper em src/cv_timeseries/models.py"
        )
