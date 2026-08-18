"""Construcao das exogenas por janela do rolling origin, com regra anti-vazamento.

Este modulo existe separado dos modelos de proposito. A politica de exogena e a peca
onde vazamento entra sem fazer barulho: um erro aqui nao quebra nada, so melhora as
metricas, e metrica que melhora sozinha ninguem investiga. Isolada aqui, ela e
testavel sem statsmodels, prophet nem timesfm, entao o teste roda em qualquer maquina
e nao tem desculpa para nao rodar.

Ver tests/test_exog.py, que exercita a invariante nas 103 janelas do desenho real.
"""

from __future__ import annotations

import pandas as pd

POLITICAS = ("climatology", "lag12", "observed")


def build_exog_frames(
    exog: pd.DataFrame,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta exogenas de treino e de futuro para uma janela do rolling origin.

    Regra anti-vazamento: para climatology/lag12, o futuro e construido a partir
    da exogena truncada no fim do treino, nunca do CSV completo.

    A politica `observed` viola isso DE PROPOSITO: e o cenario-teto de "previsao
    meteorologica perfeita", que serve para limitar quanto uma exogena futura poderia
    render. Nunca usar em resultado reportado como operacional.
    """
    exog_train = exog.loc[train_index]
    truncated = exog.loc[: train_index[-1]]

    if policy == "climatology":
        monthly = truncated.groupby(truncated.index.month).mean()
        exog_future = pd.DataFrame(
            [monthly.loc[d.month] for d in test_index], index=test_index
        )
    elif policy == "lag12":
        # Com horizonte > 12 a defasagem de 12 meses alcancaria datas posteriores ao
        # fim do treino, o que seria vazamento. Falha explicitamente em vez de deixar
        # o KeyError cru do pandas, que nao diz qual e o problema.
        needed = [d - pd.DateOffset(months=12) for d in test_index]
        missing = [d for d in needed if d not in truncated.index]
        if missing:
            raise ValueError(
                f"Politica lag12 exige o valor de 12 meses antes de cada data prevista, "
                f"mas {len(missing)} nao estao no treino (ex: {missing[0]:%Y-%m}). "
                f"Com horizonte {len(test_index)} maior que 12 a defasagem alcancaria o "
                f"futuro. Use --exog-policy climatology."
            )
        exog_future = pd.DataFrame([truncated.loc[d] for d in needed], index=test_index)
    elif policy == "observed":
        # Vazamento deliberado e rotulado: cenario-teto de "previsao meteorologica perfeita".
        exog_future = exog.loc[test_index]
    else:
        raise ValueError(f"Politica de exogena desconhecida: {policy}")

    return exog_train, exog_future
