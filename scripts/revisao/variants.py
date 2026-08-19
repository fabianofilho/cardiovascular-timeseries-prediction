"""Variantes de engenharia de features para os modelos de boosting.

Mantém EXATAMENTE o mesmo protocolo de backtest do repositorio original
(rolling_origin_splits, horizonte 6, min_train 60, janela expansiva), mudando
uma coisa de cada vez para isolar o efeito.

base      : lags 1-12, recursivo, alvo em nivel bruto  -> replica o paper
roll      : base + estatisticas de janela (media/std/min/max em 3,6,12)
cal       : base + calendario (Fourier ordem 2 sobre mes-do-ano)
diff12    : base com alvo em diferenca sazonal (y_t - y_{t-12}), reconstruido
full      : diff12 + roll + cal
direct    : full, mas multi-step direto (um modelo por horizonte) em vez de recursivo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

CAT_PARAMS = dict(
    iterations=300,
    depth=4,
    learning_rate=0.05,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
)


def build_regressor(kind: str):
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**XGB_PARAMS)
    if kind == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(**CAT_PARAMS)
    raise ValueError(kind)


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fourier de ordem 2 sobre o mes-do-ano.

    Deterministico: depende apenas da data, nunca do alvo. Nao ha vazamento
    possivel, o valor futuro e conhecido no momento da previsao.
    """
    m = index.month.to_numpy(dtype=float)
    ang = 2.0 * np.pi * m / 12.0
    return pd.DataFrame(
        {
            "sin1": np.sin(ang),
            "cos1": np.cos(ang),
            "sin2": np.sin(2 * ang),
            "cos2": np.cos(2 * ang),
        },
        index=index,
    )


def _window_features():
    from skforecast.preprocessing import RollingFeatures

    return RollingFeatures(
        stats=["mean", "std", "min", "max", "mean", "mean"],
        window_sizes=[3, 3, 3, 3, 6, 12],
    )


def _as_monthly(train: pd.Series) -> pd.Series:
    y = train.copy()
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index)
    if y.index.freq is None:
        y = y.asfreq(pd.infer_freq(y.index) or "MS")
    return y


def _future_index(y: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + y.index.freq, periods=horizon, freq=y.index.freq)


def forecast_variant(
    train: pd.Series,
    horizon: int,
    kind: str,
    variant: str,
    lags: int = 12,
) -> np.ndarray:
    """Devolve a previsao de `horizon` passos para uma variante."""
    from skforecast.recursive import ForecasterRecursive

    y = _as_monthly(train)
    fut_idx = _future_index(y, horizon)

    use_roll = variant in ("roll", "full")
    use_cal = variant in ("cal", "full", "diffcal", "direct")
    use_diff = variant in ("diff12", "full", "diffcal", "direct")

    # Alvo: nivel bruto ou diferenca sazonal.
    if use_diff:
        target = (y - y.shift(12)).dropna()
    else:
        target = y

    usable_lags = max(1, min(lags, len(target) - 1))

    exog_train = calendar_exog(target.index) if use_cal else None
    exog_future = calendar_exog(fut_idx) if use_cal else None
    window_features = _window_features() if use_roll else None

    if variant == "direct":
        from skforecast.direct import ForecasterDirect

        forecaster = ForecasterDirect(
            build_regressor(kind),
            steps=horizon,
            lags=usable_lags,
            window_features=window_features,
        )
    else:
        forecaster = ForecasterRecursive(
            build_regressor(kind),
            lags=usable_lags,
            window_features=window_features,
        )

    forecaster.fit(y=target, exog=exog_train)
    pred = forecaster.predict(steps=horizon, exog=exog_future)
    pred = np.asarray(pred, dtype=float)

    # Reconstrucao do nivel: y_{T+h} = z_{T+h} + y_{T+h-12}.
    # Com horizon <= 12, y_{T+h-12} esta sempre dentro do treino observado.
    if use_diff:
        base = np.asarray(
            [y.loc[d - pd.DateOffset(months=12)] for d in fut_idx], dtype=float
        )
        pred = pred + base

    return pred
