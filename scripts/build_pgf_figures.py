#!/usr/bin/env python3
"""Reconstroi as figuras do manuscrito como codigo pgfplots.

Motivo: figura binaria nao entra num .tex unico. Gerada como codigo, ela vira vetorial,
herda a fonte do documento, fica editavel no editor online e dispensa upload.

Le as MESMAS fontes que scripts/build_paper_assets.py: paper/verified_numbers.json para
os agregados e os CSVs de results/ para a serie. Nenhum numero e digitado aqui.

Saida: paper/figures_pgf/*.tex, cada um um tikzpicture pronto para \\input, mais um
documento de teste que compila os seis juntos.

Estas figuras sao uma ALTERNATIVA disponivel, nao as do manuscrito. O manuscrito
segue apontando para paper/figures/*.pdf. Trocar exige antes rodar a comparacao
(--compare) e olhar as seis: reconstrucao que muda a leitura do grafico e regressao,
nao melhoria. Na primeira geracao a comparacao pegou tres, todas invisiveis para o
compilador: eixo em notacao cientifica, rotulo cortado na borda e marcador com cor
errada.

Uso:
    python scripts/build_pgf_figures.py
    cd paper/figures_pgf && tectonic -X compile teste.tex
    python scripts/build_pgf_figures.py --compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIGS = ["fig1_serie", "fig2_smape_ic", "fig3_horizonte",
        "fig4_janela", "fig5_sazonal", "fig6_temperatura"]


def compara() -> int:
    """Empilha cada figura atual sobre a reconstruida, para inspecao visual."""
    import pymupdf
    from PIL import Image, ImageDraw

    teste = OUT / "teste.pdf"
    if not teste.is_file():
        raise SystemExit(f"[ERRO] {teste} nao existe. Compile teste.tex primeiro:\n"
                         f"  cd paper/figures_pgf && tectonic -X compile teste.tex")
    doc = pymupdf.open(teste)
    if doc.page_count != len(FIGS):
        raise SystemExit(f"[ERRO] teste.pdf tem {doc.page_count} paginas, esperado "
                         f"{len(FIGS)}: uma figura por pagina e pre-requisito da comparacao")

    cmp_dir = OUT / "_comparacao"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    def recorta(im):
        bb = im.convert("L").point(lambda p: 0 if p < 250 else 255).getbbox()
        if not bb:
            return im
        return im.crop((max(0, bb[0] - 12), max(0, bb[1] - 12),
                        min(im.width, bb[2] + 12), min(im.height, bb[3] + 12)))

    print("Comparando atual (matplotlib) contra reconstruida (pgfplots)")
    for i, nome in enumerate(FIGS):
        pix = doc[i].get_pixmap(dpi=150)
        novo = recorta(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        orig = recorta(Image.open(ROOT / "paper/figures" / f"{nome}.png").convert("RGB"))
        L = 1080
        esc = lambda im: im.resize((L, max(1, int(im.height * L / im.width))), Image.LANCZOS)
        orig, novo = esc(orig), esc(novo)
        f = 30
        comp = Image.new("RGB", (L, orig.height + novo.height + f * 2 + 16), "white")
        d = ImageDraw.Draw(comp)
        d.rectangle([0, 0, L, f], fill="#2c3e50")
        d.text((10, 9), f"ATUAL (matplotlib, binario)   {nome}.pdf", fill="white")
        comp.paste(orig, (0, f))
        y = f + orig.height + 16
        d.rectangle([0, y, L, y + f], fill="#7d3c98")
        d.text((10, y + 9), f"NOVO (pgfplots, codigo LaTeX)   {nome}.tex", fill="white")
        comp.paste(novo, (0, y + f))
        comp.save(cmp_dir / f"cmp_{nome}.png")
        kb = (OUT / f"{nome}.tex").stat().st_size / 1024
        kbp = (ROOT / "paper/figures" / f"{nome}.pdf").stat().st_size / 1024
        print(f"  cmp_{nome}.png   tex={kb:5.1f} KB  vs  pdf={kbp:6.1f} KB")
    print(f"\n  {len(FIGS)} comparacoes em paper/figures_pgf/_comparacao/")
    return 0

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures_pgf"
AVISO = "% gerado por scripts/build_pgf_figures.py -- nao editar a mao\n"

ROTULO = {"prophet": "Prophet", "sarima": "SARIMA", "timesfm": "TimesFM",
          "xgboost": "XGBoost", "catboost": "CatBoost"}
# mesmas cores da versao matplotlib, para a comparacao ser de forma e nao de paleta
COR = {"prophet": "4C72B0", "sarima": "DD8452", "timesfm": "55A868",
       "xgboost": "C44E52", "catboost": "8172B3"}
ORDEM = ["prophet", "sarima", "timesfm", "catboost", "xgboost"]
TOP3 = ["prophet", "sarima", "timesfm"]


def escreve(nome: str, corpo: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{nome}.tex").write_text(AVISO + corpo, encoding="utf-8")
    print(f"  paper/figures_pgf/{nome}.tex")


def defs_cor() -> str:
    return "\n".join(f"\\definecolor{{c{k}}}{{HTML}}{{{v}}}" for k, v in COR.items())


def coords(pares) -> str:
    return " ".join(f"({x},{y})" for x, y in pares)


def main() -> int:
    V = json.loads((ROOT / "paper" / "verified_numbers.json").read_text(encoding="utf-8"))
    T1, T4, T5, D = V["tabela1"], V["tabela4"], V["tabela5"], V["dados"]
    serie = pd.read_csv(RES / "series/serie_eventos_sp_sim_real_2010_2023.csv",
                        parse_dates=["date"]).set_index("date")["value"]
    temp = pd.read_csv(RES / "series/temperatura_sp_mensal_2010_2023.csv",
                       parse_dates=["date"]).set_index("date")
    hor = pd.read_csv(RES / "uncertainty_2010_2023_error_by_horizon.csv")

    print("Gerando figuras em pgfplots")

    # ---------- fig1: serie ----------
    pts = coords([(f"{d.year + (d.month - 1) / 12:.4f}", f"{v:.0f}") for d, v in serie.items()])
    escreve("fig1_serie", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.95\textwidth, height=5.0cm,
  xlabel={{}}, ylabel={{Deaths per month}},
  xmin=2009.8, xmax=2024.2, ymin=5400, ymax=11100,
  xtick={{2010,2012,2014,2016,2018,2020,2022,2024}},
  xticklabel style={{/pgf/number format/1000 sep=}},
  % sem isto o pgfplots troca o eixo por \cdot 10^4, que e ilegivel aqui
  scaled y ticks=false,
  ytick={{6000,7000,8000,9000,10000,11000}},
  yticklabel style={{/pgf/number format/fixed, /pgf/number format/1000 sep={{,}}}},
  axis lines=left, tick align=outside, tick pos=left,
  every axis plot/.append style={{line width=0.5pt}},
]
\addplot[draw=black!80, mark=none] coordinates {{{pts}}};
\addplot[draw=none, fill=cprophet, fill opacity=0.10, forget plot]
  coordinates {{(2021.0,5400) (2024.0,5400) (2024.0,11100) (2021.0,11100)}} \closedcycle;
% ancorado a direita e dentro do limite do eixo: centralizado em 2022.5 o rotulo
% estourava a borda e saia cortado
\node[anchor=east, font=\scriptsize, text=cprophet!80!black]
  at (axis cs:2023.9,10500) {{test window shared with the earlier round}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig2: sMAPE com IC ----------
    linhas = []
    for i, m in enumerate(ORDEM):
        y = len(ORDEM) - i
        d = T1[m]
        linhas.append(
            f"\\addplot[draw=c{m}, line width=2.0pt, mark=none] "
            f"coordinates {{({d['ic_low']:.4f},{y}) ({d['ic_high']:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.2pt, draw=c{m}, fill=c{m}] "
            f"coordinates {{({d['smape']:.4f},{y})}};")
    p3 = [T1[m]["smape"] for m in TOP3]
    amp, larg = V["derivados"]["amplitude_top3"], V["derivados"]["largura_media_top3"]
    ticks = ",".join(str(len(ORDEM) - i) for i in range(len(ORDEM)))
    labs = ",".join(ROTULO[m] for m in ORDEM)
    escreve("fig2_smape_ic", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.2cm,
  xlabel={{sMAPE (\%)}}, xmin=4.0, xmax=7.8,
  xtick={{4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1,
                     /pgf/number format/zerofill}},
  ymin=0.4, ymax=5.6, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Spread among the leading three: {amp:.2f} pp; mean interval width: {larg:.2f} pp}},
]
\addplot[draw=none, fill=black, fill opacity=0.14, forget plot]
  coordinates {{({min(p3):.4f},0.4) ({max(p3):.4f},0.4) ({max(p3):.4f},5.6) ({min(p3):.4f},5.6)}}
  \closedcycle;
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig3: erro por horizonte ----------
    series_h, leg = [], []
    for m in ORDEM:
        g = hor[hor.model == m].sort_values("horizon")
        series_h.append(f"\\addplot[draw=c{m}, mark=*, mark size=1.6pt, line width=0.9pt] "
                        f"coordinates {{{coords(zip(g.horizon, [f'{v:.4f}' for v in g.smape]))}}};")
        leg.append(ROTULO[m])
    escreve("fig3_horizonte", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.2cm,
  xlabel={{Forecast horizon (months)}}, ylabel={{sMAPE (\%)}},
  xmin=0.7, xmax=6.3, xtick={{1,2,3,4,5,6}},
  axis lines=left, tick align=outside, tick pos=left,
  legend style={{at={{(0.02,0.98)}}, anchor=north west, draw=none, fill=none,
                 font=\scriptsize, legend columns=3, column sep=4pt}},
  ymax=8.6,
]
{chr(10).join(series_h)}
\legend{{{','.join(leg)}}}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig4: expanding vs deslizante ----------
    linhas = []
    for i, m in enumerate(ORDEM):
        y = len(ORDEM) - i
        e, s = T4[m]["expanding"], T4[m]["deslizante"]
        dif = s - e
        cor = "C44E52" if dif > 0 else "55A868"
        linhas.append(
            f"\\addplot[draw=black!28, line width=1.2pt, mark=none] "
            f"coordinates {{({e:.4f},{y}) ({s:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw=cprophet, fill=cprophet] "
            f"coordinates {{({e:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw=csarima, fill=csarima] "
            f"coordinates {{({s:.4f},{y})}};\n"
            f"\\definecolor{{d{i}}}{{HTML}}{{{cor}}}\n"
            f"\\node[anchor=west, font=\\scriptsize, text=d{i}] "
            f"at (axis cs:{max(e, s) + 0.10:.4f},{y}) {{{dif:+.2f} pp}};")
    ticks = ",".join(str(len(ORDEM) - i) for i in range(len(ORDEM)))
    labs = ",".join(ROTULO[m] for m in ORDEM)
    escreve("fig4_janela", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.4cm,
  xlabel={{sMAPE (\%)}}, xmin=4.3, xmax=7.7,
  xtick={{4.5,5.0,5.5,6.0,6.5,7.0,7.5}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1,
                     /pgf/number format/zerofill}},
  ymin=0.4, ymax=5.7, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Cost of discarding history beyond five years}},
]
{chr(10).join(linhas)}
\node[anchor=east, font=\scriptsize] at (axis cs:7.65,5.35)
  {{\textcolor{{cprophet}}{{$\bullet$}} Expanding \quad
    \textcolor{{csarima}}{{$\bullet$}} Sliding 60}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig5: sazonal com eixo duplo ----------
    perfil = serie.groupby(serie.index.month).mean()
    tmin = temp.groupby(temp.index.month).tmin.mean()
    cm = coords(zip(range(1, 13), [f"{v:.1f}" for v in perfil]))
    ct = coords(zip(range(1, 13), [f"{v:.2f}" for v in tmin]))
    meses = "J,F,M,A,M,J,J,A,S,O,N,D"
    escreve("fig5_sazonal", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=4.8cm,
  axis y line*=left, axis x line*=bottom,
  xmin=0.5, xmax=12.5, xtick={{1,...,12}}, xticklabels={{{meses}}},
  ylabel={{Mean deaths per month}}, ylabel style={{text=cxgboost}},
  scaled y ticks=false, ytick={{6500,7000,7500,8000}},
  yticklabel style={{text=cxgboost, /pgf/number format/fixed,
                     /pgf/number format/1000 sep={{,}}}},
  tick align=outside,
]
\addplot[draw=cxgboost, mark=*, mark size=2.0pt, line width=1.1pt]
  coordinates {{{cm}}};
\end{{axis}}
\begin{{axis}}[
  width=0.80\textwidth, height=4.8cm,
  axis y line*=right, axis x line=none,
  xmin=0.5, xmax=12.5,
  ylabel={{Mean minimum temperature (C)}}, ylabel style={{text=cprophet}},
  yticklabel style={{text=cprophet}},
  tick align=outside,
]
\addplot[draw=cprophet, mark=square*, mark size=1.7pt, line width=0.9pt, dashed]
  coordinates {{{ct}}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig6: efeito da temperatura ----------
    mods = ["sarima", "catboost", "xgboost"]
    linhas = []
    for i, m in enumerate(mods):
        y = len(mods) - i
        d = T5[m]
        linhas.append(
            f"\\addplot[draw=c{m}, line width=2.0pt, mark=none] "
            f"coordinates {{({d['ic_low']:.4f},{y}) ({d['ic_high']:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.2pt, draw=c{m}, fill=c{m}] "
            f"coordinates {{({d['ganho']:.4f},{y})}};")
    ticks = ",".join(str(len(mods) - i) for i in range(len(mods)))
    labs = ",".join(ROTULO[m] for m in mods)
    escreve("fig6_temperatura", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.78\textwidth, height=3.8cm,
  xlabel={{Reduction in sMAPE from the temperature covariate (pp)}},
  xmin=-0.06, xmax=0.60,
  ymin=0.4, ymax=3.6, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
]
\draw[black!45, dashed, line width=0.6pt]
  (axis cs:0,0.4) -- (axis cs:0,3.6);
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- documento de teste ----------
    figs = ["fig1_serie", "fig2_smape_ic", "fig3_horizonte",
            "fig4_janela", "fig5_sazonal", "fig6_temperatura"]
    # Sem float e com quebra explicita: `figure[p]` empacota varias por pagina, e o
    # documento de teste existe justamente para inspecionar uma de cada vez.
    corpo = "\n\\newpage\n".join(
        f"\\noindent\\ttfamily\\small {f.replace('_', chr(92) + '_')}\\normalfont\n\n"
        f"\\vspace{{0.5em}}\n\\noindent\\input{{{f}}}" for f in figs)
    (OUT / "teste.tex").write_text(rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.8in]{{geometry}}
\usepackage{{pgfplots}}
\usepackage{{xcolor}}
\usepackage{{caption}}
\pgfplotsset{{compat=1.18}}
{defs_cor()}
\pagestyle{{empty}}
\begin{{document}}
{corpo}
\end{{document}}
""", encoding="utf-8")
    print("  paper/figures_pgf/teste.tex")
    print(f"\n  {len(figs)} figuras geradas, nenhum numero digitado a mao")
    print("  para conferir antes de trocar:")
    print("    cd paper/figures_pgf && tectonic -X compile teste.tex")
    print("    python scripts/build_pgf_figures.py --compare")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figuras do manuscrito em pgfplots")
    ap.add_argument("--compare", action="store_true",
                    help="empilha cada figura atual sobre a reconstruida, para inspecao")
    a = ap.parse_args()
    raise SystemExit(compara() if a.compare else main())
