#!/usr/bin/env python3
"""Compara as variantes de engenharia de atributos com o naive sazonal.

A pergunta nao e qual variante tem o menor sMAPE, que o CSV de metricas ja responde. E se a
melhor delas passa a barra que os boosters do paper nao passam, e se essa diferenca sobrevive
ao mesmo tratamento de incerteza usado no resto do trabalho: bootstrap pareado por janela e
teste de Diebold-Mariano, nao comparacao de ponto contra ponto.

Uso:
    PYTHONPATH=src python scripts/analisa_variantes.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
VAR = RES / "revisao" / "variants_predictions.csv"
BASE = RES / "benchmark_baselines_2010_2023_predictions.csv"
BENCH = RES / "benchmark_sim_real_sp_2010_2023_predictions.csv"
# Previsoes que chegam de fora do pipeline local, uma linha por janela e horizonte, no
# mesmo formato dos demais. Hoje e so o TabPFN, que roda por API e nao aqui. Entra como
# arquivo opcional para que a ausencia dele nao quebre a rodada de quem nao o tem, e
# para que a presenca dele nao mude nada dos outros modelos: mesmas janelas, mesma
# semente, mesmo bootstrap.
EXTRA = sorted((RES / "revisao").glob("*_predictions_externas.csv")) + [
    RES / "revisao" / "tabpfn_predictions.csv"]

B = 10_000
SEED = 20260817
REF = "snaive"          # a barra a ser batida
HORIZONTES = 6


def smape_vec(yt, yp):
    return 200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))


def matriz(df, modelo):
    g = df[df.model == modelo].sort_values(["window", "horizon"])
    nw, nh = g.window.nunique(), g.horizon.nunique()
    if len(g) != nw * nh:
        raise ValueError(f"{modelo}: {len(g)} previsoes, esperado {nw}x{nh}")
    return g.y_true.to_numpy().reshape(nw, nh), g.y_pred.to_numpy().reshape(nw, nh)


def dm_test(d, h):
    """Diebold-Mariano com correcao de Harvey, Leybourne e Newbold."""
    n = len(d)
    db = d.mean()
    var = np.sum((d - db) ** 2) / n
    for lag in range(1, h):
        var += 2.0 * np.sum((d[lag:] - db) * (d[:-lag] - db)) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = db / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def main() -> int:
    var = pd.read_csv(VAR)
    base = pd.read_csv(BASE)
    bench = pd.read_csv(BENCH)

    # Denominador do MASE: MAE em amostra do naive sazonal sobre a serie inteira.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cv_timeseries.data import load_and_aggregate_series
    from cv_timeseries.evaluate import mase_denominador
    serie = load_and_aggregate_series(
        str(RES / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"), "date", "value", "MS")
    den = mase_denominador(serie, m=12)

    quadros = [var, base, bench]
    externos = []
    for p in EXTRA:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        quadros.append(d)
        externos += sorted(d.model.unique())
        print(f"  {p.name}: {len(d)} linhas, modelos {sorted(d.model.unique())}")

    todos = pd.concat(quadros, ignore_index=True)
    nomes = sorted(var.model.unique()) + externos + [REF, "snaive_drift", "naive"]

    ae, sm = {}, {}
    for m in nomes:
        yt, yp = matriz(todos, m)
        ae[m] = np.abs(yp - yt)
        sm[m] = smape_vec(yt, yp)
    nw = ae[REF].shape[0]

    # Bootstrap pareado: as MESMAS janelas reamostradas para todos os modelos.
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nw, size=(B, nw))

    print(f"  {nw} janelas, denominador do MASE = {den:.4f}\n")
    print(f"  {'modelo':<18}{'sMAPE':>8}{'MAE':>8}{'MASE':>7}"
          f"{'vs snaive (pp)':>16}{'IC 95%':>20}{'DM sig':>8}")

    ref_sm = sm[REF]
    linhas = {}
    for m in nomes:
        dif = sm[m][idx].mean(axis=(1, 2)) - ref_sm[idx].mean(axis=(1, 2))
        lo, hi = np.percentile(dif, [2.5, 97.5])
        delta = float(sm[m].mean() - ref_sm.mean())

        # DM por horizonte, perda = erro absoluto, contra o naive sazonal.
        sig = 0
        for h in range(HORIZONTES):
            d = ae[m][:, h] - ae[REF][:, h]
            _, p = dm_test(d, h + 1)
            if p == p and p < 0.05:
                sig += 1

        linhas[m] = {
            "smape": float(sm[m].mean()), "mae": float(ae[m].mean()),
            "mase": float(ae[m].mean() / den), "delta_smape_vs_snaive": delta,
            "ic_low": float(lo), "ic_high": float(hi),
            "dm_significativos": sig, "de": HORIZONTES,
            "melhor_que_snaive": bool(hi < 0 and sig >= 3),
        }
        marca = " <-" if linhas[m]["melhor_que_snaive"] else ""
        print(f"  {m:<18}{sm[m].mean():>8.4f}{ae[m].mean():>8.1f}"
              f"{ae[m].mean() / den:>7.3f}{delta:>+16.4f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{f'{sig}/6':>8}{marca}")

    print("\n  <- criterio pre-declarado do paper: IC exclui zero E DM p<0.05 em >=3 de 6")

    # Onde o MASE do proprio naive sazonal cai. Isto e o que corrige a leitura de "MASE > 1".
    print(f"\n  MASE do naive sazonal fora da amostra: {linhas[REF]['mase']:.3f}")
    print("  Logo o ponto de comparacao NAO e 1: e este valor. O denominador do MASE e o")
    print("  erro EM AMOSTRA de um passo, e as previsoes aqui sao de um a seis passos fora.")

    # A mesma conta contra o naive sazonal COM DRIFT, que e a referencia ingenua mais
    # forte da serie. O criterio do paper compara com o naive sazonal simples, e e esse
    # que manda; esta segunda tabela existe porque o texto tambem afirma coisas sobre a
    # referencia com drift, e afirmacao sem arquivo de origem e o que este projeto passa
    # o tempo consertando.
    REF2 = "snaive_drift"
    ref2_sm = sm[REF2]
    contra_drift = {}
    for m in nomes:
        dif = sm[m][idx].mean(axis=(1, 2)) - ref2_sm[idx].mean(axis=(1, 2))
        lo, hi = np.percentile(dif, [2.5, 97.5])
        sig = 0
        ps = []
        for h in range(HORIZONTES):
            _, p = dm_test(ae[m][:, h] - ae[REF2][:, h], h + 1)
            ps.append(None if p != p else float(p))
            if p == p and p < 0.05:
                sig += 1
        contra_drift[m] = {
            "delta_smape": float(sm[m].mean() - ref2_sm.mean()),
            "ic_low": float(lo), "ic_high": float(hi),
            "dm_significativos": sig, "de": HORIZONTES,
            "dm_p_por_horizonte": ps,
            "melhor_que_snaive_drift": bool(hi < 0 and sig >= 3),
        }

    saida = RES / "revisao" / "variants_vs_snaive.json"
    saida.write_text(json.dumps({
        "_meta": {"B": B, "seed": SEED, "referencia": REF,
                  "referencia_secundaria": REF2,
                  "mase_denominador": float(den), "n_janelas": int(nw),
                  "criterio": "IC do bootstrap pareado exclui zero E DM p<0.05 em >=3 de 6"},
        "modelos": linhas,
        "modelos_vs_snaive_drift": contra_drift,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results/revisao/{saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
