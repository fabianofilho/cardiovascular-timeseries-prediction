# -*- coding: utf-8 -*-
"""Controle da tabela do termo de calendario: a linha "base" reproduz a Tabela 5?

run_temp_melhorado.py roda, na especificacao "base", exatamente o modelo da Tabela 5. Entao
a coluna sem temperatura tem que dar o sMAPE da Tabela 1, e a com temperatura o da rodada
com climatologia. Se nao der, a tabela de calendario esta noutro ambiente que o resto do
paper, que e o defeito que motivou este controle: em 28/09 a tabela ainda trazia o XGBoost
de antes de 23/09 (6,832 contra 6,880), e num Mac com as mesmas versoes deu 6,833.

Sai com codigo 1 se algum dos quatro valores divergir, para o `make regen-calendario`
parar antes de regenerar tabela e figura.

Uso:
    PYTHONPATH=src python scripts/revisao/check_controle_calendario.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "results"
TOL = 1e-6

calendario = pd.read_csv(RES / "revisao" / "temp_melhorado.csv")
sem_t = pd.read_csv(RES / "benchmark_sim_real_sp_2010_2023_metrics.csv").set_index("model").smape
com_t = pd.read_csv(RES / "benchmark_exog_temp_climatology_metrics.csv").set_index("model").smape

falhas = 0
for modelo in ("catboost", "xgboost"):
    base = calendario[(calendario.modelo == modelo) & (calendario.espec == "base")].iloc[0]
    for rotulo, obtido, esperado in (
        ("sem temperatura", base.sem_T, sem_t[modelo]),
        ("com temperatura", base.com_T, com_t[f"{modelo}_temp"]),
    ):
        dif = abs(obtido - esperado)
        ok = dif < TOL
        falhas += not ok
        print(f"  {modelo:<9}{rotulo:<17}{obtido:>11.6f}  esperado {esperado:.6f}  "
              f"{'ok' if ok else f'DIVERGE {dif:.6f}'}")

if falhas:
    print("\n  CONTROLE FALHOU: este ambiente nao e o dos numeros publicados. Nada foi "
          "regenerado. Desfaca com: git checkout results/revisao/temp_melhorado.csv")
    sys.exit(1)
print("\n  controle ok: a tabela de calendario esta no mesmo ambiente da Tabela 5")
