#!/usr/bin/env python3
"""Tabelas dos experimentos da revisao que ate agora so tinham figura.

Sao as contrapartes tabulares de revisao_fig_optuna e revisao_fig_temp_diffcal: a
figura mostra a ordem de grandeza, a tabela da os digitos e o orcamento de busca, que
e o que um revisor pede para julgar se o ajuste foi suficiente.

Mesmo modelo de scripts/build_paper_assets.py: nenhum numero digitado, tudo lido dos
JSONs e CSVs de resultado, saida em paper/tables/*.tex.

Fontes, todas em results/revisao/:
    ceiling_{catboost,xgboost}.json          teto com vazamento (oracle) e base
    optuna_{catboost,xgboost}_dev_base.json  ajuste honesto (dev)
    temp_melhorado.csv                       ganho de temperatura, base vs diffcal
    revisao_numbers.json                     referencia do naive sazonal

Uso:
    python scripts/revisao/build_revisao_tabs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import (  # noqa: E402
    ROTULO, RES, TAB, num, sgn,
)

REV = RES / "revisao"
AVISO = "% gerado por scripts/revisao/build_revisao_tabs.py -- nao editar a mao\n"


def escreve_tabela(nome, corpo):
    """Igual a do gerador original, so troca o aviso de autoria do arquivo."""
    (TAB / f"{nome}.tex").write_text(AVISO + corpo, encoding="utf-8")
    print(f"  tabela  paper/tables/{nome}.tex")
NUMBERS = json.loads((REV / "revisao_numbers.json").read_text(encoding="utf-8"))
SNAIVE = NUMBERS["tabela1_estendida"]["snaive"]["smape"]


def carrega(nome):
    return json.loads((REV / nome).read_text(encoding="utf-8"))


def tab_optuna():
    """Ajuste de hiperparametros: quanto a busca honesta rende, e quanto renderia o
    melhor caso impossivel.

    As duas colunas de ganho respondem a objecao usual de que os boosters perderam
    por falta de ajuste. A coluna dev e o que uma busca legitima entrega. A coluna
    oracle escolhe os hiperparametros olhando o proprio conjunto de teste: nao e
    desempenho, e o limite superior do que qualquer busca poderia dar. Nem esse limite
    alcanca os tres modelos lideres.
    """
    fontes = {
        "catboost": (carrega("ceiling_catboost.json")["catboost"],
                     carrega("optuna_catboost_dev_base.json")),
        "xgboost": (carrega("ceiling_xgboost.json")["xgboost"],
                    carrega("optuna_xgboost_dev_base.json")),
    }
    linhas = []
    for m, (ceil, dev) in fontes.items():
        # Base = o que ESTE ambiente produziu sem ajuste, nao o valor da Tabela 1. As
        # colunas dev e oracle vieram daqui; comparar contra um base de outro ambiente
        # misturaria o efeito do ajuste com a diferenca de versao do XGBoost.
        b = ceil["baseline_reproduzida"]
        d, o = dev["smape_benchmark"], ceil["ceiling_smape"]
        linhas.append(
            f"{ROTULO[m]} & {num(b, 2)} & {num(d, 2)} & {num(o, 2)} & "
            f"{sgn(b - d, 2)} & {sgn(b - o, 2)} & "
            f"{dev['trials']} + {ceil['trials_teto']} & "
            f"{(dev['seconds'] + ceil['segundos']) / 3600:.1f} \\\\")
    escreve_tabela("revisao_tab_optuna", rf"""\begin{{table}}[htbp]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{Hyperparameter search for the two gradient boosting models: the accuracy of the
untuned configuration reported in Table~\ref{{tab:desempenho}}, of an honest search, and of
a search allowed to see the test set. A positive gain means the search improved on the
untuned model.}}
\label{{tab:optuna}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
& \multicolumn{{3}}{{c}}{{sMAPE (\%)}} & \multicolumn{{2}}{{c}}{{Gain (pp)}} & & \\
\cmidrule(lr){{2-4}} \cmidrule(lr){{5-6}}
Model & Untuned & Honest & Leaked & Honest & Leaked & Trials & Hours \\
\midrule
{chr(10).join(linhas)}
\midrule
Seasonal naive & \multicolumn{{3}}{{c}}{{{num(SNAIVE, 2)}}} & --- & --- & --- & --- \\
\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\begin{{minipage}}{{\textwidth}}
\footnotesize
The honest column tunes on a development split carved out of the training data of each
origin, so no observation from the scored period is visible to the search; it is the column
that describes what tuning is worth. The leaked column selects the configuration by its
score on the test forecasts themselves. That is not a result, it is a bound: no search
without a time machine can do better. Both models were given the same budget, reported in
the last two columns as honest trials plus leaked trials and total wall clock. The honest
search buys CatBoost less than two tenths of a percentage point and costs XGBoost two
tenths, and neither tuned model reaches the seasonal naive method of {num(SNAIVE, 2)}\%.
Even the leaked bound leaves both more than a percentage point behind the three leading
models of Table~\ref{{tab:desempenho}}. Under-tuning is therefore not the explanation for
the boosting results. The untuned column is the value this environment produces, which for
XGBoost is {num(fontes["xgboost"][0]["baseline_reproduzida"], 2)}\% rather than the
{num(fontes["xgboost"][0]["baseline_esperada"], 2)}\% of the original run: XGBoost does not
reproduce across library versions, a limitation already declared in the manuscript. All
three columns of a row come from the same environment, so the gains are unaffected.
\end{{minipage}}
\end{{table}}""")


def tab_temp_diffcal():
    """O ganho da temperatura sobrevive a um modelo que ja enxerga a sazonalidade?

    Se a temperatura estivesse entrando como um proxy do mes do ano, o ganho deveria
    encolher ao dar aos boosters um termo de calendario e a diferenca sazonal
    explicitos. Encolhe.

    Nota: a linha 'lags only' do CatBoost bate com a Tabela 5 do artigo; a do XGBoost
    nao (6.83 aqui, 6.94 la), pela nao-reprodutibilidade do XGBoost por falta de pin de
    versao ja registrada no manuscrito. A comparacao que sustenta a conclusao e sem T
    contra com T DENTRO da mesma linha, no mesmo ambiente, e essa nao e afetada.
    """
    df = pd.read_csv(REV / "temp_melhorado.csv")
    rot = {"base": "lags only", "diffcal": "+ calendar and seasonal difference"}
    linhas = []
    for m in ("catboost", "xgboost"):
        for espec in ("base", "diffcal"):
            r = df[(df.modelo == m) & (df.espec == espec)].iloc[0]
            linhas.append(
                f"{ROTULO[m]}, {rot[espec]} & {num(r.sem_T, 2)} & {num(r.com_T, 2)} & "
                f"{sgn(r.ganho, 3)} \\\\")
        if m == "catboost":
            linhas.append("\\midrule")
    encolhe = []
    for m in ("catboost", "xgboost"):
        g = {e: float(df[(df.modelo == m) & (df.espec == e)].ganho.iloc[0])
             for e in ("base", "diffcal")}
        # em modo matematico: fora dele o sinal negativo sairia como hifen
        encolhe.append(f"{ROTULO[m]} from {sgn(g['base'], 3)} to {sgn(g['diffcal'], 3)}")
    escreve_tabela("revisao_tab_temp_diffcal", rf"""\begin{{table}}[htbp]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
\caption{{Whether the temperature gain of Table~\ref{{tab:temperatura}} survives once the
boosting models are given the annual cycle explicitly. Gain is positive when the covariate
helps.}}
\label{{tab:tempdiffcal}}
\begin{{tabular}}{{lrrr}}
\toprule
Model and feature set & Without temp. & With temp. & Gain (pp) \\
\midrule
{chr(10).join(linhas)}
\bottomrule
\end{{tabular}}

\vspace{{0.5em}}
\begin{{minipage}}{{\textwidth}}
\footnotesize
The first row of each block is the specification of Table~\ref{{tab:temperatura}}, lagged
values only. The second adds Fourier calendar terms and takes the seasonal difference
$y_t - y_{{t-12}}$ as the target, so the model no longer has to infer the annual cycle from
the covariate. The gain shrinks in both models, {encolhe[0]} and {encolhe[1]} percentage
points, and changes sign for XGBoost. The reading is that a large part of what minimum
temperature contributed in Table~\ref{{tab:temperatura}} was the month of the year rather
than the weather, and that the remaining contribution is too small to separate from noise.
The comparison that carries this conclusion is within a row, the same run with and without
the covariate; the without-temperature column of the XGBoost blocks is not identical to
Table~\ref{{tab:temperatura}} because XGBoost does not reproduce across library versions,
a limitation already declared in the manuscript.
\end{{minipage}}
\end{{table}}""")


def main() -> int:
    tab_optuna()
    tab_temp_diffcal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
