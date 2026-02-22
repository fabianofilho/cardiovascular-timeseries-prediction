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


def rolling_origin_splits(series: pd.Series, horizon: int, min_train_size: int):
    n = len(series)
    last_train_end = n - horizon
    for train_end in range(min_train_size, last_train_end + 1):
        train = series.iloc[:train_end]
        test = series.iloc[train_end : train_end + horizon]
        if len(test) == horizon:
            yield train, test
