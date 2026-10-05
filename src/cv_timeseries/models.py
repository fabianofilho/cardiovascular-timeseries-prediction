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
            # n_jobs=1, e nao -1. Com -1 o XGBoost usa todos os nucleos, e a ordem de soma
            # dos gradientes muda com quantos existem: a mesma versao da 6.827 em 4
            # threads, 6.834 em 8 e 6.832 em 12. Fixar em 1 e a unica escolha que nao
            # depende da maquina, inclusive num runner de CI com um nucleo. Custa tempo de
            # ajuste e compra reprodutibilidade. Ver docs/xgboost_reprodutibilidade.md.
            n_jobs=1,
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


class _LimitadorDeTaxa:
    """Espera o mínimo para não passar de N chamadas por minuto.

    A API da PriorLabs aceita 60 ajustes e 60 predições por minuto. Estourar o limite
    devolve erro, não fila, e uma rodada de 103 janelas leva mais de uma hora: perder
    tudo no minuto 50 por excesso de velocidade é caro. O teto usado é 50, com folga.
    """

    def __init__(self, por_minuto: int = 50):
        self._intervalo = 60.0 / por_minuto
        self._ultima = 0.0

    def espera(self) -> None:
        import time

        agora = time.monotonic()
        atraso = self._intervalo - (agora - self._ultima)
        if atraso > 0:
            time.sleep(atraso)
        self._ultima = time.monotonic()


_CLASSE_TABPFN = None


def _classe_tabpfn_limitada():
    """Constrói, uma vez só, o adaptador entre a API do TabPFN e o contrato sklearn.

    Construída aqui dentro, e não no topo do módulo, porque herda de `BaseEstimator`:
    o skforecast clona o regressor antes de ajustar, e clonar exige `get_params` e o
    protocolo de estimador. Herdar resolve isso sem reimplementá-lo — e importar o
    sklearn no topo quebraria o import preguiçoso que o resto do módulo mantém.

    Duas coisas entram aqui e não no regressor original: o limitador de taxa e a
    retentativa. Ambas são da via de rede, não do modelo, e por isso ficam fora do
    protocolo: a sequência de janelas, os lags e a recursão continuam sendo os mesmos
    que rodam o XGBoost e o CatBoost.
    """
    global _CLASSE_TABPFN
    if _CLASSE_TABPFN is not None:
        return _CLASSE_TABPFN

    import re
    import time

    from sklearn.base import BaseEstimator, RegressorMixin

    class _TabPFNRegressorLimitado(RegressorMixin, BaseEstimator):
        # Escada de espera para HTTP 429, em segundos. Nao e exponencial curta de
        # proposito: a rodada que estourou estava a 26 chamadas por minuto, bem abaixo
        # do teto de 60/min, e ainda assim recebeu 429 depois de cinco esperas de um
        # minuto. Isso e cota de janela longa, horaria ou diaria, e contra ela cinco
        # minutos de paciencia nao servem para nada. Somadas, estas esperas dao pouco
        # mais de uma hora, que e o tempo de uma janela horaria se reabrir.
        ESCADA = (60, 120, 300, 600, 900, 900, 900)

        def __init__(self, tentativas: int = 7, por_minuto: int = 50):
            self.tentativas = tentativas
            self.por_minuto = por_minuto

        @staticmethod
        def _e_limite_de_taxa(exc) -> bool:
            msg = str(exc).lower()
            return "429" in msg or "too many requests" in msg or "rate limit" in msg

        def _com_retentativa(self, fn, limitador):
            for tentativa in range(1, self.tentativas + 1):
                limitador.espera()
                try:
                    return fn()
                except Exception as exc:
                    # Erro que nao e de taxa sobe na hora: insistir num token invalido
                    # ou numa serie malformada so gasta tempo e esconde a causa.
                    if not self._e_limite_de_taxa(exc) or tentativa == self.tentativas:
                        raise
                    # Cota diaria tambem sobe na hora, pela mesma razao: a escada
                    # inteira nao reabre uma janela que so vira no dia seguinte.
                    if "daily" in str(exc).lower() or "resets at" in str(exc).lower():
                        print("[ERRO] cota diaria esgotada; o checkpoint esta salvo. "
                              "Rode de novo depois do reset informado acima.",
                              flush=True)
                        raise
                    espera = self.ESCADA[min(tentativa - 1, len(self.ESCADA) - 1)]
                    # O servidor sabe melhor que a escada quanto falta; se ele disser,
                    # e o que vale.
                    achado = re.search(r"retry-?after[^0-9]*(\d+)", str(exc).lower())
                    if achado:
                        espera = max(espera, float(achado.group(1)))
                    print(f"[INFO] limite de taxa; esperando {espera / 60:.0f} min "
                          f"(tentativa {tentativa}/{self.tentativas})", flush=True)
                    time.sleep(espera + 1)
            raise RuntimeError("inalcançável")

        def fit(self, X, y):
            from tabpfn_client import TabPFNRegressor

            self._modelo = TabPFNRegressor()
            self._lim_fit = _LimitadorDeTaxa(self.por_minuto)
            self._lim_pred = _LimitadorDeTaxa(self.por_minuto)
            self._com_retentativa(lambda: self._modelo.fit(X, y), self._lim_fit)
            return self

        def predict(self, X):
            return self._com_retentativa(
                lambda: self._modelo.predict(X), self._lim_pred)

    _CLASSE_TABPFN = _TabPFNRegressorLimitado
    return _CLASSE_TABPFN


class TabPFNForecaster(_SkforecastRecursiveForecaster):
    """TabPFN pela API da PriorLabs, na mesma via recursiva dos dois boosters.

    Existe para responder ao parecer sem trocar de bancada. Rodar o TabPFN por fora e
    trazer só o sMAPE agregado foi o que deixou a afirmação sobre ele sem o teste
    pareado que todas as outras afirmações do artigo têm: o critério pré-declarado
    precisa da previsão janela a janela, e ela só é comparável se sair do mesmo
    caminho que gerou as demais.

    Requer `tabpfn_client` e o token em TABPFN_TOKEN. Custo de uma rodada completa:
    103 ajustes e 618 predições na API, pouco mais de uma hora.
    """

    name = "tabpfn"
    supports_exog = False       # a via da API não recebe exógena aqui

    def __init__(self, lags: int = 12, **kwargs):
        self.lags = lags
        self._kwargs = kwargs

    def _build_regressor(self):
        return _classe_tabpfn_limitada()(**self._kwargs)


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


class NaiveForecaster(Forecaster):
    """Ultimo valor observado, repetido. A referencia mais crua que existe."""

    name = "naive"

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        return np.full(horizon, float(train.iloc[-1]))


class SeasonalNaiveForecaster(Forecaster):
    """Mesmo mes do ano anterior. A referencia que importa em serie sazonal.

    E a barra que qualquer modelo precisa passar para justificar existir: se ele nao
    vence repetir o ano passado, ele nao aprendeu sazonalidade, so a reproduziu pior.
    """

    name = "snaive"
    m = 12

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        if len(train) < self.m:
            return np.full(horizon, float(train.iloc[-1]))
        return np.array([float(train.iloc[-self.m + i % self.m]) for i in range(horizon)])


class SeasonalNaiveDriftForecaster(SeasonalNaiveForecaster):
    """Naive sazonal mais a tendencia do ultimo ano, distribuida no horizonte.

    Existe porque a serie tem tendencia de alta (79.933 obitos em 2010, 95.538 em 2023):
    sem o drift, a referencia sazonal pura subestima sistematicamente, e vencer uma
    referencia enviesada nao prova nada.
    """

    name = "snaive_drift"

    def forecast(self, train, horizon, exog_train=None, exog_future=None):
        base = super().forecast(train, horizon)
        if len(train) <= self.m + 1:
            return base
        drift = float(train.iloc[-1]) - float(train.iloc[-self.m - 1])
        return base + drift * np.arange(1, horizon + 1) / self.m
