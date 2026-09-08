# -*- coding: utf-8 -*-
"""Testa a alegacao do ponto 23: o truncamento em h-1 subestima a variancia do DM?

A alegacao: em h=1 o truncamento em h-1=0 defasagens usa apenas gamma0, sem absorver
autocorrelacao nenhuma. Mas janelas de origens vizinhas sao sobrepostas (compartilham
quase todo o treino e preveem meses consecutivos), entao o diferencial de perda d_t deve
ser serialmente correlacionado. Se for, a variancia esta subestimada e o teste rejeita
demais justamente no horizonte que mais pesa no criterio de 3 em 6.

Isso e uma alegacao teorica que eu nao tinha verificado. Este script mede.

Compara tres estimadores da variancia de longo prazo:
  (a) h-1        truncamento do paper
  (b) NW         Newey-West/Bartlett, L = floor(4*(n/100)^(2/9))
  (c) NW-h       Newey-West com L = max(h-1, floor(n^(1/3)))
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ADAPTADO ao versionar: a entrada era out/final_predictions.csv, com fallback para um
# caminho absoluto D:/ da maquina dele. Aqui os cinco modelos e as baselines moram em dois
# arquivos do proprio repositorio e sao concatenados na hora. O conteudo e o mesmo.
RAIZ = Path(__file__).resolve().parents[2]
FONTES = ("benchmark_sim_real_sp_2010_2023_predictions.csv",
          "benchmark_baselines_2010_2023_predictions.csv")

MODELOS = ["prophet", "sarima", "timesfm", "catboost", "xgboost", "snaive"]


def acf(x, nlags=6):
    x = x - x.mean()
    n = len(x)
    g0 = np.sum(x * x) / n
    return [float(np.sum(x[l:] * x[:-l]) / n / g0) for l in range(1, nlags + 1)]


def lrvar(d, L):
    """Variancia de longo prazo com pesos de Bartlett e L defasagens."""
    n = len(d)
    dm = d - d.mean()
    v = np.sum(dm * dm) / n
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        v += 2.0 * w * np.sum(dm[l:] * dm[:-l]) / n
    return v


def dm_p(d, L):
    n = len(d)
    v = lrvar(d, L)
    if v <= 0:
        return np.nan
    stat = d.mean() / np.sqrt(v / n)
    # correcao de Harvey-Leybourne-Newbold, com h = L+1 equivalente
    h = L + 1
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return float(2 * (1 - stats.t.cdf(abs(stat * corr), df=n - 1)))


def main():
    df = pd.concat([pd.read_csv(RAIZ / "results" / f) for f in FONTES],
                   ignore_index=True)   # ADAPTADO
    df = df[df.model.isin(MODELOS)]
    nw = int(df.window.max())
    ae = {}
    for m in MODELOS:
        g = df[df.model == m].sort_values(["window", "horizon"])
        ae[m] = np.abs(g.y_pred.to_numpy(float) - g.y_true.to_numpy(float)).reshape(nw, 6)

    L_nw = int(np.floor(4 * (nw / 100.0) ** (2.0 / 9.0)))
    print(f"n = {nw} janelas | L de Newey-West = {L_nw}\n")

    print("=== 1. Autocorrelacao do diferencial de perda em h=1 ===")
    print("   (se ~0, minha alegacao estava errada)")
    for a, b in [("prophet", "catboost"), ("prophet", "sarima"), ("snaive", "catboost")]:
        d = ae[a][:, 0] - ae[b][:, 0]
        r = acf(d, 5)
        print(f"   {a[:8]:>8} - {b[:8]:<9} rho1..rho5 = " + " ".join(f"{x:+.3f}" for x in r))

    print("\n=== 2. Impacto nos p-valores em h=1 ===")
    print(f"   {'par':<24}{'p (h-1, paper)':>16}{'p (NW)':>10}{'p (NW-h)':>10}  muda?")
    n_muda = 0
    total = 0
    for a, b in combinations(MODELOS, 2):
        d = ae[a][:, 0] - ae[b][:, 0]
        p0 = dm_p(d, 0)                      # truncamento h-1 = 0
        p1 = dm_p(d, L_nw)
        p2 = dm_p(d, max(0, int(nw ** (1 / 3))))
        total += 1
        muda = (p0 < 0.05) != (p1 < 0.05)
        n_muda += muda
        print(f"   {a[:10]+'-'+b[:10]:<24}{p0:16.4f}{p1:10.4f}{p2:10.4f}  "
              f"{'SIM' if muda else ''}")
    print(f"\n   celulas que mudam de veredito em h=1: {n_muda} de {total}")

    print("\n=== 3. Todos os horizontes: quantas celulas mudam ===")
    tot, muda = 0, 0
    for a, b in combinations(MODELOS, 2):
        for h in range(1, 7):
            d = ae[a][:, h - 1] - ae[b][:, h - 1]
            p0 = dm_p(d, h - 1)
            p1 = dm_p(d, max(h - 1, L_nw))
            tot += 1
            if (p0 < 0.05) != (p1 < 0.05):
                muda += 1
                print(f"   MUDA: {a}-{b} h={h}  p_paper={p0:.4f} -> p_NW={p1:.4f}")
    print(f"\n   total: {muda} de {tot} celulas mudam de veredito")

    # ADAPTADO ao versionar: grava o resultado. O script original so imprimia, e um
    # resultado que so existe no terminal nao pode ser conferido depois nem travado em
    # teste. A conta nao mudou.
    dest = RAIZ / "results" / "revisao" / "dm_variancia.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "_meta": {
            "pergunta": "o truncamento em h-1 subestima a variancia do DM?",
            "n_janelas": nw, "L_newey_west": L_nw,
            "estimadores": ["h-1 (paper)", "Newey-West/Bartlett", "NW com L=max(h-1, n^(1/3))"],
        },
        "acf_h1": {
            f"{a}-{b}": acf(ae[a][:, 0] - ae[b][:, 0], 5)
            for a, b in [("prophet", "catboost"), ("prophet", "sarima"),
                         ("snaive", "catboost")]
        },
        "celulas_que_mudam_h1": int(n_muda),
        "celulas_h1": int(total),
        "celulas_que_mudam_total": int(muda),
        "celulas_total": int(tot),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"   results/revisao/{dest.name}")


if __name__ == "__main__":
    main()
