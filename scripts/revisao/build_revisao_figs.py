#!/usr/bin/env python3
"""Figuras dos experimentos da revisao (Optuna e variantes), no mesmo modelo de
scripts/build_paper_assets.py: tikzpicture gerado por codigo a partir dos JSONs de
resultado, paleta e fonte compartilhadas, PNG via tectonic para conferencia.

Fontes:
    results/revisao/ceiling_{catboost,xgboost}.json         teto com vazamento (oracle)
    results/revisao/optuna_{catboost,xgboost}_dev_base.json ajuste honesto (dev)
    results/revisao/variants_vs_snaive.json                 12 variantes de atributos

Saidas:
    paper/figures/revisao_fig_optuna.tex (+.png)
    paper/figures/revisao_fig_variantes.tex (+.png)

Uso:
    python scripts/revisao/build_revisao_figs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import (  # noqa: E402  reusa paleta, fonte e infra do original
    COR, ROTULO, RES, escreve_figura, rasteriza_figuras, coords,
)

REV = RES / "revisao"

SNAIVE_SMAPE = 6.269313196510599  # V["tabela1"]["snaive"], build_paper_assets.py


def carrega(nome):
    return json.loads((REV / nome).read_text(encoding="utf-8"))


def fig_optuna():
    """Ajuste honesto (dev) nao supera o modelo base; o teto com vazamento (oracle)
    e o melhor caso possivel para uma busca, e mesmo assim nao bate o naive sazonal
    para o CatBoost."""
    ceil_cb = carrega("ceiling_catboost.json")["catboost"]
    ceil_xg = carrega("ceiling_xgboost.json")["xgboost"]
    dev_cb = carrega("optuna_catboost_dev_base.json")
    dev_xg = carrega("optuna_xgboost_dev_base.json")

    linhas_modelo = [
        ("catboost", ceil_cb["baseline_reproduzida"], dev_cb["smape_benchmark"],
         ceil_cb["ceiling_smape"]),
        ("xgboost", ceil_xg["baseline_esperada"], dev_xg["smape_benchmark"],
         ceil_xg["ceiling_smape"]),
    ]
    etapas = [("base", "square*", "sem ajuste"),
              ("dev", "*", "ajuste honesto (dev)"),
              ("oracle", "triangle*", "teto com vazamento (oracle)")]

    linhas, rotulos, y = [], [], 0
    n_linhas = len(linhas_modelo) * len(etapas)
    for modelo, base, dev, ceiling in linhas_modelo:
        for (chave, marca, _), valor in zip(etapas, (base, dev, ceiling)):
            y += 1
            linhas.append(
                f"\\addplot[only marks, mark={marca}, mark size=2.4pt, "
                f"draw=c{modelo}, fill=c{modelo}] "
                f"coordinates {{({valor:.4f},{n_linhas - y + 1})}};")
            rotulos.append((n_linhas - y + 1, f"{ROTULO[modelo]}, {chave}"))
    ticks = ",".join(str(t) for t, _ in rotulos)
    # cada rotulo entre chaves: sem isto pgfplots separa yticklabels pela virgula
    # que ja usamos dentro do proprio texto ("CatBoost, base") e desalinha os ticks
    labs = ",".join(f"{{{lab}}}" for _, lab in rotulos)
    escreve_figura("revisao_fig_optuna", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.82\textwidth, height=4.6cm,
  xlabel={{sMAPE (\%)}}, xmin=6.1, xmax=7.2,
  xtick={{6.2,6.4,6.6,6.8,7.0,7.2}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  ymin=0.4, ymax={n_linhas + 0.6:.1f}, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Honest tuning does not close the gap; the leaked ceiling barely does}},
  legend style={{at={{(0.98,0.98)}}, anchor=north east, draw=none, fill=none,
                 font=\scriptsize, text=black, legend columns=1}},
]
\draw[black!45, dashed, line width=0.7pt]
  (axis cs:{SNAIVE_SMAPE:.4f},0.4) -- (axis cs:{SNAIVE_SMAPE:.4f},{n_linhas + 0.6:.1f})
  node[midway, above, sloped, font=\scriptsize, text=black]
  {{seasonal naive, {SNAIVE_SMAPE:.2f}\%}};
{chr(10).join(linhas)}
\addlegendimage{{only marks, mark=square*, mark size=2.4pt, draw=black, fill=black}}
\addlegendentry{{base (no tuning)}}
\addlegendimage{{only marks, mark=*, mark size=2.4pt, draw=black, fill=black}}
\addlegendentry{{honest tuning (dev)}}
\addlegendimage{{only marks, mark=triangle*, mark size=2.8pt, draw=black, fill=black}}
\addlegendentry{{leaked ceiling (oracle)}}
\end{{axis}}
\end{{tikzpicture}}""")


def fig_variantes():
    """As 12 variantes de engenharia de atributos, como diferenca de sMAPE contra o
    seasonal naive, no mesmo modelo da Figura 6 do artigo (fig6_temperatura): reta
    pontilhada em zero, ponto + IC de bootstrap por linha."""
    d = carrega("variants_vs_snaive.json")["modelos"]
    rotulo_variante = {
        "base": "lags only (base)", "cal": "+ calendar Fourier",
        "diff12": "+ seasonal difference", "direct": "direct strategy",
        "full": "full (cal+diff12+direct)", "roll": "+ rolling stats",
    }
    ordem_variante = ["direct", "diff12", "full", "cal", "base", "roll"]
    linhas_modelo = [("catboost", ordem_variante), ("xgboost", ordem_variante)]

    linhas, rotulos, y = [], [], 0
    total = sum(len(v) for _, v in linhas_modelo)
    for modelo, variantes in linhas_modelo:
        for var in variantes:
            y += 1
            g = d[f"{modelo}_{var}"]
            yy = total - y + 1
            linhas.append(
                f"\\addplot[draw=c{modelo}, line width=2.0pt, mark=none] "
                f"coordinates {{({g['ic_low']:.4f},{yy}) ({g['ic_high']:.4f},{yy})}};\n"
                f"\\addplot[only marks, mark=*, mark size=2.2pt, draw=c{modelo}, "
                f"fill=c{modelo}] coordinates {{({g['delta_smape_vs_snaive']:.4f},{yy})}};")
            rotulos.append((yy, f"{ROTULO[modelo]}, {rotulo_variante[var]}"))
    ticks = ",".join(str(t) for t, _ in rotulos)
    labs = ",".join(f"{{{lab}}}" for _, lab in rotulos)
    escreve_figura("revisao_fig_variantes", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.85\textwidth, height=7.2cm,
  xlabel={{sMAPE difference from seasonal naive (pp)}},
  xmin=-0.55, xmax=1.20,
  xtick={{-0.5,-0.25,0,0.25,0.5,0.75,1.0}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=2}},
  ymin=0.4, ymax={total + 0.6:.1f}, ytick={{{ticks}}}, yticklabels={{{labs}}},
  y tick label style={{font=\scriptsize}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{None of the twelve feature variants beats seasonal naive at the
    pre-declared threshold}},
]
\draw[black!45, dashed, line width=0.7pt]
  (axis cs:0,0.4) -- (axis cs:0,{total + 0.6:.1f});
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")


def main() -> int:
    fig_optuna()
    fig_variantes()
    rasteriza_figuras()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
