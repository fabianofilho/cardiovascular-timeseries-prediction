from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str
    supports_exog: bool = False

    @abstractmethod
    def forecast(
        self,
        train: pd.Series,
        horizon: int,
        exog_train: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class SarimaForecaster(Forecaster):
    name = "sarima"
    supports_exog = True

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        # Import preguicoso, como nos outros cinco modelos deste arquivo. Ansioso no
        # topo, ele derrubava o modulo inteiro quando statsmodels faltava, e com ele
        # qualquer execucao, mesmo uma que so pedisse timesfm. Aqui a falta so atinge
        # quem instancia o SARIMA, e o build_models consegue avisar e seguir.
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._sarimax_cls = SARIMAX
        self.order = order
        self.seasonal_order = seasonal_order

    def forecast(
        self,
        train: pd.Series,
        horizon: int,
        exog_train: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> np.ndarray:
        model = self._sarimax_cls(
            train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fit = model.fit(disp=False, maxiter=200)
        pred = fit.forecast(steps=horizon, exog=exog_future)
        return np.asarray(pred, dtype=float)


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self):
        from prophet import Prophet

        self._prophet_cls = Prophet

    def forecast(
        self,
        train: pd.Series,
        horizon: int,
        exog_train: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> np.ndarray:
        # Exógenas não são usadas pelo Prophet neste benchmark.
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


class _SkforecastRecursiveForecaster(Forecaster):
    """Base para modelos de boosting via skforecast ForecasterRecursive.

    Usa previsão recursiva multi-passo com features de lag. Subclasses
    fornecem o regressor sklearn-compatível em `_build_regressor`.
    """

    name: str
    lags: int = 12
    supports_exog = True

    def _build_regressor(self):  # pragma: no cover - implementado nas subclasses
        raise NotImplementedError

    def forecast(
        self,
        train: pd.Series,
        horizon: int,
        exog_train: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> np.ndarray:
        from skforecast.recursive import ForecasterRecursive

        # skforecast exige índice datetime com frequência definida.
        y = train.copy()
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index)
        if y.index.freq is None:
            inferred = pd.infer_freq(y.index)
            y = y.asfreq(inferred or "MS")

        if exog_train is not None:
            exog_train = exog_train.copy()
            exog_train.index = y.index
        if exog_future is not None:
            exog_future = exog_future.copy()
            exog_future.index = pd.date_range(
                y.index[-1] + y.index.freq, periods=horizon, freq=y.index.freq
            )

        # Não usar mais lags do que o histórico permite.
        usable_lags = max(1, min(self.lags, len(y) - 1))

        # Posicional para compatibilidade: o 1º argumento chama-se `regressor`
        # até skforecast 0.16 e `estimator` a partir de versões mais novas.
        forecaster = ForecasterRecursive(
            self._build_regressor(),
            lags=usable_lags,
        )
        forecaster.fit(y=y, exog=exog_train)
        pred = forecaster.predict(steps=horizon, exog=exog_future)
        return np.asarray(pred, dtype=float)


class XGBoostForecaster(_SkforecastRecursiveForecaster):
    name = "xgboost"

    def __init__(self, lags: int = 12, **xgb_kwargs):
        self.lags = lags
        self._xgb_kwargs = xgb_kwargs

    def _build_regressor(self):
        from xgboost import XGBRegressor

        params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
        params.update(self._xgb_kwargs)
        return XGBRegressor(**params)


class CatBoostForecaster(_SkforecastRecursiveForecaster):
    name = "catboost"

    def __init__(self, lags: int = 12, **cat_kwargs):
        self.lags = lags
        self._cat_kwargs = cat_kwargs

    def _build_regressor(self):
        from catboost import CatBoostRegressor

        params = dict(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        params.update(self._cat_kwargs)
        return CatBoostRegressor(**params)


class TimesFMForecaster(Forecaster):
    name = "timesfm"

    def __init__(self, repo_id: str = "google/timesfm-2.5-200m-pytorch"):
        import timesfm

        self._timesfm = timesfm
        self._model = None
        self._model_horizon = None
        self._repo_id = repo_id

    def _build_model(self, horizon: int):
        # API nova (TimesFM 2.5)
        if hasattr(self._timesfm, "TimesFM_2p5_200M_torch") and hasattr(
            self._timesfm, "ForecastConfig"
        ):
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            model = self._timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self._repo_id,
                token=hf_token,
            )
            model.compile(
                self._timesfm.ForecastConfig(
                    max_context=512,
                    max_horizon=horizon,
                    normalize_inputs=True,
                )
            )
            return model

        # API antiga
        if hasattr(self._timesfm, "TimesFmHparams") and hasattr(
            self._timesfm, "TimesFmCheckpoint"
        ):
            hparams = self._timesfm.TimesFmHparams(
                backend="cpu",
                context_len=512,
                horizon_len=horizon,
                per_core_batch_size=1,
            )
            checkpoint = self._timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            )
            return self._timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

        raise RuntimeError("API TimesFMHparams/TimesFMCheckpoint não encontrada.")

    def forecast(
        self,
        train: pd.Series,
        horizon: int,
        exog_train: pd.DataFrame | None = None,
        exog_future: pd.DataFrame | None = None,
    ) -> np.ndarray:
        # Exógenas não são usadas pelo TimesFM zero-shot.
        values = train.to_numpy(dtype=float)

        # Recarrega se o horizonte mudar entre execuções.
        if self._model is None or self._model_horizon != horizon:
            self._model = self._build_model(horizon=horizon)
            self._model_horizon = horizon

        if hasattr(self._model, "forecast"):
            # API nova: forecast(horizon=..., inputs=[...])
            try:
                preds, _ = self._model.forecast(horizon=horizon, inputs=[values])
                return np.asarray(preds[0][:horizon], dtype=float)
            except TypeError:
                # API antiga: forecast([values], freq=[0])
                preds, _ = self._model.forecast([values], freq=[0])
                return np.asarray(preds[0][:horizon], dtype=float)

        raise RuntimeError(
            "Interface TimesFM não reconhecida para esta versão. "
            "Ajuste o wrapper em src/cv_timeseries/models.py"
        )
