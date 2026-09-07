#!/usr/bin/env python3
"""A temperatura ajuda o Prophet?

O manuscrito diz que Prophet nao aceita covariavel exogena e por isso ficou de fora da
Tabela 5. Aceita, por `add_regressor()`. Este script mede o que o paper deixou de medir,
com a MESMA politica climatology sem vazamento e o MESMO criterio de decisao das outras
comparacoes: bootstrap pareado por janela mais Diebold-Mariano por horizonte.

A comparacao e Prophet contra ele mesmo, com e sem a covariavel, entao a checagem que
importa e a de PROCEDENCIA: a rodada sem temperatura tem que reproduzir a linha do Prophet
na Tabela 1. Sem isso, um eventual efeito da temperatura poderia ser so a diferenca entre
duas implementacoes de Prophet.

Uso:
    PYTHONPATH=src python scripts/analisa_prophet_temp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRED = RES / "revisao" / "prophet_exog_predictions.csv"
BENCH = RES / "benchmark_sim_real_sp_2010_2023_predictions.csv"

B = 10_000
SEED = 20260817
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
    d = pd.read_csv(PRED)
    ae, sm = {}, {}
    for m in ("prophet_repro", "prophet_temp", "prophet_temp_ceiling"):
        yt, yp = matriz(d, m)
        ae[m] = np.abs(yp - yt)
        sm[m] = smape_vec(yt, yp)
    nw = ae["prophet_repro"].shape[0]

    # --- procedencia: a rodada sem temperatura reproduz a Tabela 1? ---------
    b = pd.read_csv(BENCH)
    b = b[b.model == "prophet"][["window", "horizon", "y_pred"]]
    g = d[d.model == "prophet_repro"][["window", "horizon", "y_pred"]]
    j = g.merge(b, on=["window", "horizon"], suffixes=("", "_bench"))
    div = float(np.abs(j.y_pred - j.y_pred_bench).max())
    print(f"  Procedencia: {len(j)} previsoes, divergencia maxima contra a Tabela 1 "
          f"= {div:.2e}  -> {'identica' if div < 1e-6 else 'DIVERGE'}")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nw, size=(B, nw))
    ref = sm["prophet_repro"]

    print(f"\n  {'rodada':<22}{'sMAPE':>8}{'ganho (pp)':>13}{'IC 95%':>22}{'DM sig':>9}")
    out = {}
    for m in ("prophet_repro", "prophet_temp", "prophet_temp_ceiling"):
        dif = sm[m][idx].mean(axis=(1, 2)) - ref[idx].mean(axis=(1, 2))
        lo, hi = np.percentile(dif, [2.5, 97.5])
        # ganho positivo = a temperatura MELHOROU, seguindo a convencao da Tabela 5
        ganho = float(ref.mean() - sm[m].mean())
        sig = 0
        for h in range(HORIZONTES):
            _, p = dm_test(ae[m][:, h] - ae["prophet_repro"][:, h], h + 1)
            if p == p and p < 0.05:
                sig += 1
        out[m] = {"smape": float(sm[m].mean()), "mae": float(ae[m].mean()),
                  "ganho_pp": ganho, "ic_low": float(-hi), "ic_high": float(-lo),
                  "dm_significativos": sig, "de": HORIZONTES}
        print(f"  {m:<22}{sm[m].mean():>8.4f}{ganho:>+13.4f}"
              f"{f'[{-hi:+.3f}, {-lo:+.3f}]':>22}{f'{sig}/6':>9}")

    t = out["prophet_temp"]
    veredito = ("a temperatura PIORA o Prophet" if t["ganho_pp"] < 0
                else "a temperatura melhora o Prophet")
    passa = t["ic_low"] > 0 and t["dm_significativos"] >= 3
    print(f"\n  Ponto: {veredito} em {abs(t['ganho_pp']):.3f} pp.")
    print(f"  Criterio pre-declarado: {'ATENDIDO' if passa else 'NAO atendido'} "
          f"(IC contem zero: {t['ic_low'] < 0 < t['ic_high']}, "
          f"DM {t['dm_significativos']}/6).")

    saida = RES / "revisao" / "prophet_temp_vs_base.json"
    saida.write_text(json.dumps({
        "_meta": {"B": B, "seed": SEED, "n_janelas": int(nw),
                  "politica": "climatology (sem vazamento); ceiling = observed (vazamento)",
                  "procedencia_max_div": div,
                  "criterio": "IC do bootstrap pareado exclui zero E DM p<0.05 em >=3 de 6"},
        "modelos": out,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results/revisao/{saida.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
