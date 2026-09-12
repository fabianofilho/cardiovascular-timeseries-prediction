#!/usr/bin/env python3
"""Segunda leva de figuras da revisao: as analises que ainda nao tinham figura, mais
o teste de encolhimento do ganho de temperatura (falsificabilidade do ponto 21).

Fontes, todas em results/revisao/ (repo) ou copiadas do Drive para la:
    revisao_numbers.json      tabela1_estendida (R1, baselines ingenuas) e
                               calibracao (R6, PICP/interval score por horizonte) e
                               tabela4_enriched (janela expansiva vs deslizante, direct)
    temp_melhorado.csv        ganho de temperatura, spec base vs spec diffcal (R_temp)

Mesmo modelo de scripts/build_paper_assets.py: tikzpicture por codigo, paleta e fonte
compartilhadas via import, PNG via tectonic para conferencia.

Uso:
    python scripts/revisao/build_revisao_figs2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import (  # noqa: E402
    ROTULO, RES, escreve_figura, rasteriza_figuras,
)

REV = RES / "revisao"
NUMBERS = json.loads((REV / "revisao_numbers.json").read_text(encoding="utf-8"))

ROTULO_EXT = dict(ROTULO)  # copia; nao poluir o dict importado


def fig_naive_estendida():
    """R1: a Figura 2 do artigo (sMAPE com IC) com os tres benchmarks ingenuos
    acrescentados, no mesmo eixo. Mostra o seasonal naive dentro da faixa do
    boosting, o achado central de R1."""
    ordem = ["prophet", "sarima", "timesfm", "catboost", "xgboost",
             "snaive_drift", "snaive", "naive"]
    t1 = NUMBERS["tabela1_estendida"]
    n = len(ordem)
    linhas = []
    for i, m in enumerate(ordem):
        y = n - i
        d = t1[m]
        cor = f"c{m}" if m in ("prophet", "sarima", "timesfm", "catboost", "xgboost") \
            else "black!55"
        linhas.append(
            f"\\addplot[draw={cor}, line width=2.0pt, mark=none] "
            f"coordinates {{({d['ic_low']:.4f},{y}) ({d['ic_high']:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.2pt, draw={cor}, fill={cor}] "
            f"coordinates {{({d['smape']:.4f},{y})}};")
    p3 = [t1[m]["smape"] for m in ("prophet", "sarima", "timesfm")]
    ticks = ",".join(str(n - i) for i in range(n))
    labs = ",".join(f"{{{ROTULO_EXT[m]}}}" for m in ordem)
    escreve_figura("revisao_fig_naive_estendida", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.82\textwidth, height=6.6cm,
  xlabel={{sMAPE (\%)}}, xmin=4.0, xmax=11.8,
  xtick={{4,5,6,7,8,9,10,11}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=0}},
  ymin=0.4, ymax={n + 0.6:.1f}, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Seasonal naive sits inside the boosting range}},
]
\addplot[draw=none, fill=black, fill opacity=0.14, forget plot]
  coordinates {{({min(p3):.4f},0.4) ({max(p3):.4f},0.4) ({max(p3):.4f},{n + 0.6:.1f})
                ({min(p3):.4f},{n + 0.6:.1f})}}
  \closedcycle;
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_calibracao():
    """R6: cobertura empirica por horizonte, SARIMA e Prophet, contra o nominal
    de 95%. E a evidencia de que o Prophet, lider em sMAPE, tem o pior intervalo."""
    cal = NUMBERS["calibracao"]
    series = []
    for m, cor in (("sarima", "csarima"), ("prophet", "cprophet")):
        pts = cal[m]["coverage"]["por_horizonte"]
        series.append(
            f"\\addplot[draw={cor}, mark=*, mark size=1.8pt, line width=1.1pt, "
            f"mark options={{draw={cor}, fill={cor}}}] coordinates {{"
            + " ".join(f"({h},{v:.4f})" for h, v in enumerate(pts, 1)) + "};")
    escreve_figura("revisao_fig_calibracao", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.0cm,
  xlabel={{Forecast horizon (months)}}, ylabel={{Empirical coverage}},
  xmin=0.7, xmax=6.3, xtick={{1,2,3,4,5,6}},
  ymin=0.70, ymax=1.00, ytick={{0.70,0.75,0.80,0.85,0.90,0.95,1.00}},
  yticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=2}},
  axis lines=left, tick align=outside, tick pos=left,
  legend style={{at={{(0.02,0.06)}}, anchor=south west, draw=none, fill=none,
                 font=\small, text=black}},
]
\draw[black!45, dashed, line width=0.9pt]
  (axis cs:0.7,0.95) -- (axis cs:6.3,0.95)
  node[midway, above, font=\scriptsize, text=black] {{95\% nominal}};
{chr(10).join(series)}
\legend{{SARIMA,Prophet}}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_enriched_janela():
    """Janela expansiva vs deslizante-60 para as variantes enriquecidas (direct),
    no mesmo modelo da Figura 2 do artigo (fig4_janela): a vantagem da engenharia
    de atributos tambem nao sobrevive a historia curta, igual SARIMA e Prophet."""
    t4 = NUMBERS["tabela4_enriched"]
    modelos = ["CatBoost, enriched", "XGBoost, enriched"]
    n = len(modelos)
    linhas = []
    for i, rot in enumerate(modelos):
        y = n - i
        d = t4[rot]
        e, s = d["expanding"], d["sliding60"]
        dif = s - e
        cor = "ccatboost" if "CatBoost" in rot else "cxgboost"
        linhas.append(
            f"\\addplot[draw=black!28, line width=1.2pt, mark=none] "
            f"coordinates {{({e:.4f},{y}) ({s:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw={cor}, fill={cor}] "
            f"coordinates {{({e:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=square*, mark size=2.0pt, draw=black!55, "
            f"fill=black!55] coordinates {{({s:.4f},{y})}};\n"
            f"\\node[anchor=west, font=\\scriptsize, text=black] "
            f"at (axis cs:{max(e, s) + 0.10:.4f},{y}) {{{dif:+.2f} pp}};")
    ticks = ",".join(str(n - i) for i in range(n))
    labs = ",".join(f"{{{m}}}" for m in modelos)
    escreve_figura("revisao_fig_enriched_janela", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.82\textwidth, height=3.6cm,
  xlabel={{sMAPE (\%)}}, xmin=5.9, xmax=7.2,
  xtick={{6.0,6.2,6.4,6.6,6.8,7.0,7.2}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  ymin=0.4, ymax={n + 0.6:.1f}, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{The enriched variants also lose their edge under short history}},
]
{chr(10).join(linhas)}
\node[anchor=east, font=\scriptsize, text=black] at (axis cs:7.15,{n + 0.35:.1f})
  {{\textcolor{{black!55}}{{$\bullet$}} Expanding \quad
    \textcolor{{black!55}}{{$\blacksquare$}} Sliding 60}};
\end{{axis}}
\end{{tikzpicture}}""")


def fig_temp_diffcal():
    """Teste de falsificabilidade (ponto 21): o ganho de temperatura encolhe quando
    os boosters recebem um termo de calendario e diferenca sazonal explicitos?

    Nota metodologica: a linha 'base, sem T' de CatBoost aqui bate com a Tabela 5
    do artigo (6.589, ambiente deterministico). A de XGBoost NAO bate (6.827 aqui
    contra 6.935 na Tabela 5) -- e a mesma nao-reprodutibilidade do XGBoost por
    falta de pin de versao ja registrada no manuscrito (6.83 vs 6.94), nao um erro
    deste script. A comparacao valida aqui e DENTRO da mesma linha (sem T vs com T,
    mesmo ambiente), que e o que testa a hipotese; a pasta kaggle/ (checada) nao
    contem dados de temperatura, so os kernels do teto do Optuna.
    """
    df = pd.read_csv(REV / "temp_melhorado.csv")
    ordem = [("catboost", "base"), ("catboost", "diffcal"),
             ("xgboost", "base"), ("xgboost", "diffcal")]
    rot_espec = {"base": "lags only (base)", "diffcal": "+ calendar \\& seasonal diff"}
    n = len(ordem)
    linhas, labs_rows = [], []
    for i, (modelo, espec) in enumerate(ordem):
        y = n - i
        row = df[(df.modelo == modelo) & (df.espec == espec)].iloc[0]
        cor = f"c{modelo}"
        linhas.append(
            f"\\addplot[draw=black!28, line width=1.2pt, mark=none] "
            f"coordinates {{({row.com_T:.4f},{y}) ({row.sem_T:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=square*, mark size=2.2pt, draw={cor}, "
            f"fill=white, line width=0.9pt] coordinates {{({row.sem_T:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw={cor}, fill={cor}] "
            f"coordinates {{({row.com_T:.4f},{y})}};\n"
            f"\\node[anchor=west, font=\\scriptsize, text=black] "
            f"at (axis cs:{max(row.sem_T, row.com_T) + 0.06:.4f},{y}) "
            f"{{{row.ganho:+.2f} pp}};")
        labs_rows.append((y, f"{ROTULO_EXT[modelo]}, {rot_espec[espec]}"))
    ticks = ",".join(str(t) for t, _ in labs_rows)
    labs = ",".join(f"{{{lab}}}" for _, lab in labs_rows)
    escreve_figura("revisao_fig_temp_diffcal", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.84\textwidth, height=4.0cm,
  xlabel={{sMAPE (\%)}}, xmin=5.9, xmax=7.1,
  xtick={{6.0,6.2,6.4,6.6,6.8,7.0}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  ymin=0.4, ymax={n + 1.1:.1f}, ytick={{{ticks}}}, yticklabels={{{labs}}},
  y tick label style={{font=\scriptsize}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{The temperature gain shrinks once boosters get explicit seasonality}},
]
{chr(10).join(linhas)}
\node[anchor=north west, font=\scriptsize, text=black]
  at (axis cs:5.92,{n + 1.0:.1f})
  {{$\square$ without temp. \quad $\bullet$ with temp.}};
\end{{axis}}
\end{{tikzpicture}}""")


def main() -> int:
    fig_naive_estendida()
    fig_calibracao()
    fig_enriched_janela()
    fig_temp_diffcal()
    rasteriza_figuras()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
