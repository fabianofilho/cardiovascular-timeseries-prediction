"""Reconstrói a série observada e o índice janela/horizonte do benchmark do SIM.

Contexto: `data/processed/serie_eventos_sp_sim_real.csv` e `data/raw/sim_real_sp_latest.csv`
não estão versionados (ver LAB-62), então um clone limpo não consegue reproduzir nem auditar
o benchmark publicado. O que está versionado é
`results/benchmark_sim_real_sp_2019_2023_predictions.csv`, com a coluna `y_true` de cada
ponto de teste.

Este script recupera dali:

1. a série observada do período efetivamente testado, 2021-01 a 2023-12, 36 meses;
2. o índice (janela, horizonte) de cada previsão, que o CSV original não traz explicitamente.

Limite importante: os 24 primeiros meses da série (2019-01 a 2020-12) nunca entraram em
janela de teste, só de treino, então NÃO são recuperáveis por aqui. A série reconstruída tem
36 dos 60 meses originais e não substitui a extração oficial do SIM.

O índice de janela e horizonte é deduzido da ordem das linhas: o benchmark grava as previsões
em blocos contíguos de 6, um bloco por origem do rolling origin. O script valida essa premissa
em vez de confiar nela, exigindo que cada bloco seja de 6 meses consecutivos.

Uso:
    python scripts/rebuild_series_from_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = ROOT / "results" / "benchmark_sim_real_sp_2019_2023_predictions.csv"
METRICS_CSV = ROOT / "results" / "benchmark_sim_real_sp_2019_2023_metrics.csv"
OUT_SERIES = ROOT / "data" / "processed" / "serie_sim_sp_2021_2023_reconstruida.csv"
OUT_INDEX = ROOT / "results" / "predictions_indexed.csv"

HORIZON = 6


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE em porcentagem, denominador (|y| + |yhat|) / 2."""
    return float(
        np.mean(200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def index_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Atribui (window, horizon) por blocos contíguos de HORIZON linhas, validando a premissa."""
    out = []
    for model, g in df.groupby("model", sort=True):
        g = g.reset_index(drop=True)
        if len(g) % HORIZON != 0:
            raise SystemExit(f"{model}: {len(g)} linhas não são múltiplo de {HORIZON}")
        g["window"] = g.index // HORIZON + 1
        g["horizon"] = g.index % HORIZON + 1
        for w, blk in g.groupby("window"):
            meses = blk["date"].dt.year * 12 + blk["date"].dt.month
            if not np.array_equal(np.diff(meses.to_numpy()), np.ones(HORIZON - 1)):
                raise SystemExit(
                    f"{model}, janela {w}: datas não são 6 meses consecutivos, "
                    "a premissa de blocos contíguos não vale para este arquivo"
                )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> int:
    df = pd.read_csv(PRED_CSV, parse_dates=["date"])
    df = index_windows(df)

    # y_true precisa ser único por data, senão a série observada não é bem definida
    por_data = df.groupby("date")["y_true"].nunique()
    if (por_data > 1).any():
        raise SystemExit("y_true diverge entre modelos na mesma data, série não reconstruível")

    serie = (
        df.groupby("date", as_index=False)["y_true"]
        .first()
        .rename(columns={"y_true": "value"})
        .sort_values("date")
    )
    serie["origem"] = "reconstruida_de_y_true_do_benchmark"

    # teste de sanidade: as métricas recalculadas têm de bater com as publicadas
    oficial = pd.read_csv(METRICS_CSV).set_index("model")
    linhas = []
    ok = True
    for model, g in df.groupby("model", sort=True):
        yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
        calc = {"mae": mae(yt, yp), "rmse": rmse(yt, yp), "smape": smape(yt, yp)}
        for met, val in calc.items():
            ref = float(oficial.loc[model, met])
            bate = abs(val - ref) < 0.005
            ok &= bate
            linhas.append((model, met, ref, val, "ok" if bate else "DIVERGE"))

    print(f"Janelas por modelo: {df.groupby('model')['window'].max().to_dict()}")
    print(f"Série reconstruída: {len(serie)} meses, "
          f"{serie['date'].min():%Y-%m} a {serie['date'].max():%Y-%m}")
    print(f"Média mensal: {serie['value'].mean():.1f} óbitos, "
          f"mín {serie['value'].min():.0f}, máx {serie['value'].max():.0f}")
    print("\nConferência contra as métricas publicadas:")
    print(f"{'modelo':<10}{'métrica':<8}{'publicado':>12}{'recalculado':>14}  status")
    for model, met, ref, val, status in linhas:
        print(f"{model:<10}{met:<8}{ref:>12.4f}{val:>14.4f}  {status}")

    if not ok:
        print("\nFALHA: recálculo não bate com o CSV publicado. Nada foi gravado.")
        return 1

    OUT_SERIES.parent.mkdir(parents=True, exist_ok=True)
    serie.to_csv(OUT_SERIES, index=False)
    df.to_csv(OUT_INDEX, index=False)
    print(f"\nGravado: {OUT_SERIES.relative_to(ROOT)}")
    print(f"Gravado: {OUT_INDEX.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
