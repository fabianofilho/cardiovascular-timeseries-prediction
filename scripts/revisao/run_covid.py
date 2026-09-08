#!/usr/bin/env python3
"""Analise de sensibilidade ao periodo COVID-19.

O manuscrito lista a pandemia como limitacao mas nao roda nenhuma sensibilidade. A serie
de teste (2015-01 a 2023-12) contem 2020-2022, com choques de nivel que podem dominar as
comparacoes entre modelos.

Desenho: exclui JANELAS INTEIRAS cujo periodo de teste toca o periodo pandemico, em vez de
excluir datas soltas. Isso mantem a estrutura retangular (janela x horizonte) que o
bootstrap pareado exige. Duas definicoes, porque a escolha do recorte e discutivel:

  agudo : 2020-03 a 2021-12   (fase aguda, ate o fim da vacinacao em massa)
  amplo : 2020-01 a 2022-12   (inclui Omicron e o excesso de mortalidade tardio)

Reaproveita as previsoes ja geradas: nenhum modelo e reajustado, entao o unico efeito
medido e o da composicao do periodo de teste.
"""
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd

RECORTES = {
    "completo": None,
    "sem_covid_agudo": ("2020-03", "2021-12"),
    "sem_covid_amplo": ("2020-01", "2022-12"),
}
MODELOS = ["prophet", "sarima", "timesfm", "snaive", "catboost", "xgboost"]
B, SEED = 10_000, 20260817


def smape_vec(a, b):
    return 200.0 * np.abs(b - a) / (np.abs(a) + np.abs(b))


def main():
    # ADAPTADO ao versionar: no Drive havia um out/final_predictions.csv com tudo junto.
    # Aqui os cinco modelos e as baselines ingenuas moram em dois arquivos distintos, e a
    # concatenacao e feita na hora. O conteudo e o mesmo.
    raiz = Path(__file__).resolve().parents[2]
    partes = [
        pd.read_csv(raiz / "results" / f, parse_dates=["date"])
        for f in ("benchmark_sim_real_sp_2010_2023_predictions.csv",
                  "benchmark_baselines_2010_2023_predictions.csv")
    ]
    d = pd.concat(partes, ignore_index=True)
    d = d[d.model.isin(MODELOS)].copy()
    faltando = set(MODELOS) - set(d.model.unique())
    if faltando:
        raise SystemExit(f"modelos ausentes nas previsoes: {sorted(faltando)}")

    for nome, faixa in RECORTES.items():
        if faixa is None:
            keep = sorted(d.window.unique())
        else:
            lo, hi = pd.Period(faixa[0], "M"), pd.Period(faixa[1], "M")
            per = d.date.dt.to_period("M")
            ruins = set(d.loc[(per >= lo) & (per <= hi), "window"].unique())
            keep = sorted(set(d.window.unique()) - ruins)

        sub = d[d.window.isin(keep)]
        nw, nh = len(keep), int(sub.horizon.max())
        print(f"\n=== {nome}: {nw} janelas ({nw*nh} previsoes por modelo) ===")
        if nw < 10:
            print("  janelas insuficientes, recorte ignorado"); continue

        sm = {}
        for m in MODELOS:
            g = sub[sub.model == m].sort_values(["window", "horizon"])
            sm[m] = smape_vec(g.y_true.to_numpy(float), g.y_pred.to_numpy(float)).reshape(nw, nh)

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, nw, size=(B, nw))
        boot = {m: sm[m][idx].mean(axis=(1, 2)) for m in MODELOS}

        print(f"  {'modelo':<10}{'sMAPE':>8}{'IC95':>20}")
        for m in sorted(MODELOS, key=lambda x: sm[x].mean()):
            lo_, hi_ = np.percentile(boot[m], [2.5, 97.5])
            print(f"  {m:<10}{sm[m].mean():8.3f}   [{lo_:5.3f}, {hi_:5.3f}]")

        print("  pares entre os lideres:")
        for a, b_ in combinations(["prophet", "sarima", "timesfm"], 2):
            dif = sm[a].mean() - sm[b_].mean()
            bd = boot[a] - boot[b_]
            l, h = np.percentile(bd, [2.5, 97.5])
            p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
            print(f"    {a}-{b_:<9} {dif:+.3f}  IC[{l:+.3f},{h:+.3f}]  p={min(p,1):.3f}"
                  f"  {'contem zero' if l <= 0 <= h else 'EXCLUI ZERO'}")
        print("  boosting vs seasonal naive:")
        for m in ["catboost", "xgboost"]:
            bd = boot["snaive"] - boot[m]
            l, h = np.percentile(bd, [2.5, 97.5])
            p = 2 * min((bd >= 0).mean(), (bd <= 0).mean())
            print(f"    snaive-{m:<9} {sm['snaive'].mean()-sm[m].mean():+.3f}  "
                  f"IC[{l:+.3f},{h:+.3f}]  p={min(p,1):.3f}")


if __name__ == "__main__":
    main()
