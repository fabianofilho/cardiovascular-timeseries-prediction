#!/usr/bin/env python3
"""Sensibilidade ao periodo COVID, sob o criterio completo do paper.

`scripts/revisao/run_covid.py` monta os recortes e aplica o bootstrap pareado. Falta a outra
metade do criterio pre-declarado, que exige tambem Diebold-Mariano com p<0,05 em ao menos 3
de 6 horizontes. Este script aplica as duas.

Desenho, herdado do script original: exclui JANELAS INTEIRAS cujo periodo de teste toca a
pandemia, e nao datas soltas. Isso preserva a estrutura retangular (janela x horizonte) que o
bootstrap pareado exige. Nenhum modelo e reajustado, entao o unico efeito medido e o da
composicao do periodo de teste.

O recorte "completo" existe como CONTROLE: tem que reproduzir a Tabela 1. Se nao reproduzir,
os outros dois recortes nao significam nada.

Uso:
    PYTHONPATH=src python scripts/analisa_covid.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

RECORTES = {
    "completo": None,
    "sem_covid_agudo": ("2020-03", "2021-12"),
    "sem_covid_amplo": ("2020-01", "2022-12"),
}
TOP3 = ["prophet", "sarima", "timesfm"]
BOOST = ["catboost", "xgboost"]
MODELOS = TOP3 + BOOST + ["snaive"]
B, SEED = 10_000, 20260817


def smape_vec(a, b):
    return 200.0 * np.abs(b - a) / (np.abs(a) + np.abs(b))


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


def carrega():
    partes = [
        pd.read_csv(RES / f, parse_dates=["date"])
        for f in ("benchmark_sim_real_sp_2010_2023_predictions.csv",
                  "benchmark_baselines_2010_2023_predictions.csv")
    ]
    d = pd.concat(partes, ignore_index=True)
    return d[d.model.isin(MODELOS)].copy()


def janelas_do_recorte(d, faixa):
    if faixa is None:
        return sorted(d.window.unique())
    lo, hi = pd.Period(faixa[0], "M"), pd.Period(faixa[1], "M")
    per = d.date.dt.to_period("M")
    ruins = set(d.loc[(per >= lo) & (per <= hi), "window"].unique())
    return sorted(set(d.window.unique()) - ruins)


def main() -> int:
    d = carrega()
    verificados = json.loads((ROOT / "paper" / "verified_numbers.json")
                             .read_text(encoding="utf-8"))["tabela1"]
    saida = {"_meta": {"B": B, "seed": SEED,
                       "desenho": "exclui janelas inteiras cujo teste toca o periodo",
                       "criterio": "IC do bootstrap pareado exclui zero E DM p<0.05 em >=3 de 6"},
             "recortes": {}}

    for nome, faixa in RECORTES.items():
        keep = janelas_do_recorte(d, faixa)
        sub = d[d.window.isin(keep)]
        nw, nh = len(keep), int(sub.horizon.max())
        print(f"\n=== {nome}: {nw} janelas, {nw * nh} previsoes por modelo ===")

        sm, ae = {}, {}
        for m in MODELOS:
            g = sub[sub.model == m].sort_values(["window", "horizon"])
            yt = g.y_true.to_numpy(float).reshape(nw, nh)
            yp = g.y_pred.to_numpy(float).reshape(nw, nh)
            sm[m] = smape_vec(yt, yp)
            ae[m] = np.abs(yp - yt)

        # controle: o recorte completo tem que reproduzir a Tabela 1
        if faixa is None:
            div = max(abs(float(sm[m].mean()) - verificados[m]["smape"])
                      for m in MODELOS if m in verificados)
            print(f"  controle: divergencia maxima contra a Tabela 1 = {div:.2e}"
                  f"  -> {'reproduz' if div < 1e-6 else 'DIVERGE'}")
            saida["_meta"]["controle_max_div"] = div

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, nw, size=(B, nw))
        boot = {m: sm[m][idx].mean(axis=(1, 2)) for m in MODELOS}

        rec = {"n_janelas": nw, "smape": {m: float(sm[m].mean()) for m in MODELOS},
               "pares_top3": {}, "boosting_vs_snaive": {}}

        print("  pares entre os tres lideres (o achado central do paper):")
        for a, b in combinations(TOP3, 2):
            bd = boot[a] - boot[b]
            lo, hi = np.percentile(bd, [2.5, 97.5])
            nsig = sum(dm_test(ae[a][:, h] - ae[b][:, h], h + 1)[1] < 0.05
                       for h in range(nh))
            passa = not (lo <= 0 <= hi) and nsig >= 3
            rec["pares_top3"][f"{a}-{b}"] = {
                "delta": float(sm[a].mean() - sm[b].mean()),
                "ic_low": float(lo), "ic_high": float(hi),
                "dm_significativos": int(nsig), "distinguiveis": bool(passa)}
            print(f"    {a}-{b:<9} {sm[a].mean() - sm[b].mean():+.3f}  "
                  f"IC[{lo:+.3f},{hi:+.3f}]  DM {nsig}/6  "
                  f"-> {'DISTINGUIVEIS' if passa else 'indistinguiveis'}")

        print("  boosting contra o naive sazonal:")
        for m in BOOST:
            bd = boot[m] - boot["snaive"]
            lo, hi = np.percentile(bd, [2.5, 97.5])
            nsig = sum(dm_test(ae[m][:, h] - ae["snaive"][:, h], h + 1)[1] < 0.05
                       for h in range(nh))
            passa = lo > 0 and nsig >= 3
            rec["boosting_vs_snaive"][m] = {
                "delta": float(sm[m].mean() - sm["snaive"].mean()),
                "ic_low": float(lo), "ic_high": float(hi),
                "dm_significativos": int(nsig), "pior_que_snaive": bool(passa)}
            print(f"    {m:<9} {sm[m].mean() - sm['snaive'].mean():+.3f}  "
                  f"IC[{lo:+.3f},{hi:+.3f}]  DM {nsig}/6  "
                  f"-> {'PIOR que o naive sazonal' if passa else 'nao estabelecido'}")

        saida["recortes"][nome] = rec

    dest = RES / "revisao" / "covid_sensibilidade.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(saida, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results/revisao/{dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
