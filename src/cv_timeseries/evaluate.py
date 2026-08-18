from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    model_name: str
    mae: float
    rmse: float
    smape: float


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def rolling_origin_splits(
    series: pd.Series,
    horizon: int,
    min_train_size: int,
    max_train_size: int | None = None,
):
    """Janelas de rolling origin.

    max_train_size=None: janela expanding (treino começa no primeiro ponto).
    max_train_size=k: janela deslizante; o treino é limitado aos últimos k
    pontos antes da origem. As origens de teste são as mesmas nos dois modos,
    o que mantém o pareamento entre execuções.
    """
    n = len(series)
    last_train_end = n - horizon
    for train_end in range(min_train_size, last_train_end + 1):
        start = 0 if max_train_size is None else max(0, train_end - max_train_size)
        train = series.iloc[start:train_end]
        test = series.iloc[train_end : train_end + horizon]
        if len(test) == horizon:
            yield train, test


def mase(y_true: np.ndarray, y_pred: np.ndarray, denominador: float) -> float:
    """Mean Absolute Scaled Error: MAE dividido pelo MAE do naive sazonal em amostra.

    Metrica padrao da literatura de forecasting, e a que responde a pergunta que sMAPE
    e MAE nao respondem: o modelo vale mais que a regra ingenua? MASE < 1 vence o naive
    sazonal, MASE > 1 perde para ele.

    Um benchmark sem baseline ingenua nao diz se os modelos aprenderam algo ou apenas
    reproduziram a sazonalidade. Em revista de forecasting essa ausencia e a primeira
    coisa que o revisor cobra.
    """
    return float(mae(y_true, y_pred) / denominador)


def mase_denominador(series: pd.Series, m: int = 12) -> float:
    """MAE do naive sazonal EM AMOSTRA, o denominador canonico do MASE.

    Calculado sobre a serie inteira, nao por janela: e uma constante de escala da serie,
    para que o MASE de modelos diferentes seja comparavel entre si.
    """
    v = np.asarray(series, dtype=float)
    if len(v) <= m:
        raise ValueError(f"serie com {len(v)} pontos nao permite defasagem sazonal de {m}")
    return float(np.mean(np.abs(v[m:] - v[:-m])))
