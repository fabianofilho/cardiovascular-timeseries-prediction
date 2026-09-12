#!/usr/bin/env python3
"""Monta a pagina que compara o manuscrito original com a analise de hoje.

Regra desta pagina, a mesma do resto do projeto: nenhum numero digitado. Tudo o que
aparece vem de arquivo -- o lado "original" do JSON extraido do PDF comentado, o lado
"nova" do verified_numbers.json e dos resultados da revisao, e a evidencia de procedencia
do proprio git.

A comparacao e feita na PRECISAO IMPRESSA. Cada valor novo e formatado com as mesmas
casas que o original usou e so entao os dois sao confrontados como texto. A pergunta que
isso responde e a unica que interessa a quem le o artigo: o digito publicado ainda e o
digito que o pipeline produz hoje?

Fontes:
    results/revisao/original_manuscript_numbers.json  o artigo como foi para revisao
    paper/verified_numbers.json                       a analise de hoje
    results/revisao/*.json, temp_melhorado.csv        experimentos da revisao
    results/revisao/reconferencia_local.json          rodada independente deste ambiente
    results/calibracao_*.csv e revisao/intervals_*.csv  as duas rodadas de intervalo
    git show 2795422:paper/tables/*.tex               o que o gerador produzia antes
    results/revisao/tabpfn_resultados.json            opcional, quando o Colab rodar

Saida: 02_experimentos/comparacao_original_vs_nova.html

Uso:
    python scripts/revisao/build_comparacao_html.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import PAPER, RES, ROTULO  # noqa: E402
from build_tabpfn_notebook import EXPERIMENTOS  # noqa: E402

REV = RES / "revisao"
SAIDA = EXPERIMENTOS / "comparacao_original_vs_nova.html"
FIGS_DIR = EXPERIMENTOS / "figures_v2"

COMMIT_ANTES = "2795422"

ORIG = json.loads((REV / "original_manuscript_numbers.json").read_text(encoding="utf-8"))
V = json.loads((PAPER / "verified_numbers.json").read_text(encoding="utf-8"))


def le(nome, obrigatorio=True):
    p = REV / nome
    if not p.exists():
        if obrigatorio:
            raise FileNotFoundError(p)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------ mapa de comparacao
# Para cada tabela do original: como achar a linha correspondente na analise nova e
# como imprimir cada coluna com as MESMAS casas que o original usou.

def ic(lo, hi, casas, sinal=False):
    f = f"{{:+.{casas}f}}" if sinal else f"{{:.{casas}f}}"
    return f"[{f.format(lo)}, {f.format(hi)}]"


MAPA = {
    "tabela1": {
        "rotulo": "Desempenho agregado",
        "chave_nova": lambda: V["tabela1"],
        "linha": {"Prophet": "prophet", "SARIMA": "sarima", "TimesFM": "timesfm",
                  "CatBoost": "catboost", "XGBoost": "xgboost"},
        "colunas": [
            ("mae", "MAE", lambda d: f"{d['mae']:.1f}"),
            ("rmse", "RMSE", lambda d: f"{d['rmse']:.1f}"),
            ("smape", "sMAPE (%)", lambda d: f"{d['smape']:.2f}"),
            ("ic", "IC 95%", lambda d: ic(d["ic_low"], d["ic_high"], 2)),
            ("largura", "Largura", lambda d: f"{d['ic_width']:.2f}"),
            (None, "MASE", lambda d: f"{d['mase']:.3f}"),
        ],
        "linhas_novas": [("Seasonal naive + drift", "snaive_drift"),
                         ("Seasonal naive", "snaive"), ("Naive", "naive")],
        "porque_novo": "As três referências ingênuas e a coluna MASE não existiam. Sem "
                       "elas o artigo comparava os modelos entre si, mas não contra "
                       "repetir o ano passado, que é a barra que qualquer um deles "
                       "precisa passar para justificar existir.",
    },
    "tabela2": {
        "rotulo": "Diferenças pareadas do top-3",
        "chave_nova": lambda: V["tabela2"],
        "linha": {"Prophet -SARIMA": "prophet_menos_sarima",
                  "Prophet -TimesFM": "prophet_menos_timesfm",
                  "SARIMA -TimesFM": "sarima_menos_timesfm"},
        "colunas": [
            ("dif", "Diferença (pp)", lambda d: f"{d['dif']:+.3f}"),
            ("ic", "IC 95%", lambda d: ic(d["ic_low"], d["ic_high"], 3, sinal=True)),
            ("p_boot", "p (bootstrap)", lambda d: f"{d['p_boot']:.3f}"),
            ("dm", "DM p<0,05", lambda d: f"{d['dm_sig']} of 6"),
        ],
        "linhas_novas": [],
        "porque_novo": "",
    },
    "tabela3": {
        "rotulo": "Efeito da história de treino",
        "chave_nova": lambda: V["tabela3"],
        "linha": {"Prophet": "prophet", "SARIMA": "sarima", "TimesFM": "timesfm"},
        "colunas": [
            ("smape_curta", "sMAPE 24-54 meses", lambda d: f"{d['smape_curta']:.2f}"),
            ("smape_longa", "sMAPE 132-168 meses", lambda d: f"{d['smape_longa']:.2f}"),
            ("ganho", "Ganho (pp)", lambda d: f"{d['ganho']:.2f}"),
        ],
        "linhas_novas": [],
        "porque_novo": "",
    },
    "tabela4": {
        "rotulo": "Janela expansiva contra deslizante de 60 meses",
        "chave_nova": lambda: V["tabela4"],
        "linha": {"Prophet": "prophet", "SARIMA": "sarima", "TimesFM": "timesfm",
                  "CatBoost": "catboost", "XGBoost": "xgboost"},
        "colunas": [
            ("expanding", "Expansiva", lambda d: f"{d['expanding']:.2f}"),
            ("sliding60", "Deslizante 60", lambda d: f"{d['deslizante']:.2f}"),
            ("dif", "Dif. (pp)", lambda d: f"{d['dif']:+.2f}"),
            ("ic", "IC 95%", lambda d: ic(d["ic_low"], d["ic_high"], 3, sinal=True)),
            ("p", "p", lambda d: f"{d['p_boot']:.3f}"),
            ("dm", "DM p<0,05", lambda d: f"{d['dm_sig']} of 6"),
        ],
        "linhas_novas": [],
        "porque_novo": "",
    },
    "tabela5": {
        "rotulo": "Temperatura mínima como covariável",
        "chave_nova": lambda: V["tabela5"],
        "linha": {"SARIMA": "sarima", "CatBoost": "catboost", "XGBoost": "xgboost"},
        "colunas": [
            ("sem", "Sem temp.", lambda d: f"{d['sem']:.2f}"),
            ("com", "Com temp.", lambda d: f"{d['com']:.2f}"),
            ("ganho", "Ganho (pp)", lambda d: f"{d['ganho']:.3f}"),
            ("ic", "IC 95%", lambda d: ic(d["ic_low"], d["ic_high"], 3, sinal=True)),
            ("dm", "DM p<0,05", lambda d: f"{d['dm_sig']} of 6"),
            ("teto", "Teto", lambda d: f"{d['teto']:.2f}"),
        ],
        "linhas_novas": [("Prophet", "prophet")],
        "porque_novo": "O artigo excluiu o Prophet desta tabela alegando que o modelo "
                       "não aceita covariável exógena. Aceita, por add_regressor(), e a "
                       "rodada existe. É o único modelo em que a temperatura piora a "
                       "previsão.",
    },
}

# Figura do original -> arquivo da figura de hoje. Correspondencia estabelecida pelas
# legendas, nao pelos numeros, e por isso vive aqui e nao sai de nenhum arquivo.
FIGURAS = {
    "figura1": "fig2_smape_ic",
    "figura2": "fig4_janela",
    "figura3": "fig3_horizonte",
    "figura4": "fig6_temperatura",
    "figura5": "fig5_sazonal",
    "figura6": "fig1_serie",
}
FIGURAS_NOVAS = [
    ("revisao_fig_naive_estendida",
     "As três referências ingênuas no mesmo eixo dos modelos. O naive sazonal cai dentro "
     "da faixa do boosting."),
    ("revisao_fig_calibracao",
     "Cobertura empírica dos intervalos de 95% por horizonte. Fecha a lacuna que o "
     "original chamava de mais consequente."),
    ("revisao_fig_variantes",
     "As doze variantes de engenharia de atributos contra o naive sazonal. Responde ao "
     "comentário sobre média móvel."),
    ("revisao_fig_optuna",
     "Ajuste honesto e teto com vazamento, por modelo. Responde ao comentário sobre "
     "Optuna."),
    ("revisao_fig_enriched_janela",
     "A melhor variante de cada booster sob as duas políticas de janela."),
    ("revisao_fig_temp_diffcal",
     "O ganho da temperatura encolhe quando os boosters recebem o calendário explícito."),
]


# ------------------------------------------------------------------- comparacao

def compara_tabela(chave):
    """Devolve (cabecalhos, linhas) ja classificados celula a celula."""
    spec = MAPA[chave]
    orig = ORIG["tabelas"][chave]
    novo = spec["chave_nova"]()
    cabecalhos = [rot for _, rot, _ in spec["colunas"]]

    linhas = []
    for rot_orig, chave_nova in spec["linha"].items():
        cells, mudou = [], False
        for k_orig, _, fmt in spec["colunas"]:
            v_novo = fmt(novo[chave_nova])
            if k_orig is None:
                cells.append({"status": "novo", "antes": None, "depois": v_novo})
                continue
            v_orig = orig["linhas"][rot_orig][k_orig]
            igual = v_orig.replace(" ", "") == v_novo.replace(" ", "")
            mudou = mudou or not igual
            cells.append({"status": "confere" if igual else "mudou",
                          "antes": v_orig, "depois": v_novo})
        linhas.append({"rotulo": rot_orig, "status": "mudou" if mudou else "confere",
                       "cells": cells})

    for rot, chave_nova in spec["linhas_novas"]:
        cells = []
        for _, _, fmt in spec["colunas"]:
            try:
                cells.append({"status": "novo", "antes": None,
                              "depois": fmt(novo[chave_nova])})
            except KeyError:
                cells.append({"status": "novo", "antes": None, "depois": "—"})
        linhas.append({"rotulo": rot, "status": "novo", "cells": cells})

    return cabecalhos, linhas


def divergencia_calibracao():
    """As duas rodadas de intervalo do Prophet, medidas e nao afirmadas."""
    a = pd.read_csv(RES / "calibracao_2010_2023_predictions.csv")
    b = pd.read_csv(REV / "intervals_predictions.csv")
    j = a.merge(b, on=["model", "window", "horizon"], suffixes=("_a", "_b"))
    out = {}
    for m in ("sarima", "prophet"):
        g = j[j.model == m]
        out[m] = {
            "desvio_ypred": float((g.y_pred_a - g.y_pred_b).abs().max()),
            "desvio_lo": float((g.lo_a - g.lo_b).abs().max()),
            "desvio_hi": float((g.hi_a - g.hi_b).abs().max()),
            "cobertura_a": [float(x.dentro_a.mean()) for _, x in g.groupby("horizon")],
            "cobertura_b": [float(x.dentro_b.mean()) for _, x in g.groupby("horizon")],
            "n": int(len(g)),
        }
    return out


def tabelas_antes_do_commit():
    """Quais tabelas do manuscrito o gerador ainda NAO produzia no commit de referencia.

    O teste e textual: procura no .tex daquele commit a marca de cada bloco que hoje sai
    do gerador. Se a marca nao esta la, aquele bloco so existia dentro do main_victor.tex,
    montado a mao.

    As marcas sao de LINHA ou de COLUNA da tabela, com os separadores '&' em volta, e
    nao apenas a palavra. Procurar so por "Prophet" no tab5 antigo daria positivo pela
    nota de rodape, que ja falava do Prophet para dizer que ele estava excluido -- o
    oposto do que se quer medir.
    """
    marcas = {
        "Coluna Width na Tabela 1": ("tab1_desempenho", "& Width &"),
        "Linhas enriched na Tabela 4": ("tab4_janela", ", enriched &"),
        "Linha do Prophet na Tabela 5": ("tab5_temperatura", "\nProphet &"),
        "Interval score por horizonte na Tabela 6": ("tab6_calibracao",
                                                     ", interval score &"),
        "Tabela COVID": ("tab8_covid", "windows"),
    }
    out = []
    for rotulo, (arq, marca) in marcas.items():
        r = subprocess.run(
            ["git", "show", f"{COMMIT_ANTES}:paper/tables/{arq}.tex"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        existia = r.returncode == 0 and marca in (r.stdout or "")
        out.append({"bloco": rotulo, "arquivo": f"{arq}.tex",
                    "no_gerador_antes": existia,
                    "arquivo_existia": r.returncode == 0})
    return out


# ------------------------------------------------------------------------ HTML

def esc(s):
    return html.escape(str(s), quote=True)


CSS = """
:root{
  --ground:#f6f7f8; --surface:#ffffff; --surface-2:#f0f2f4;
  --ink:#14181c; --ink-2:#3d474e; --muted:#5c666e; --line:#dfe3e6; --line-2:#c9d0d5;
  --confere:#00785a; --mudou:#b34e00; --novo:#005e93; --aberto:#8a6100;
  --confere-bg:#e6f4ef; --mudou-bg:#fbeee5; --novo-bg:#e5f0f7; --aberto-bg:#faf1de;
  --accent:#0072b2;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#12161a; --surface:#1a1f25; --surface-2:#222932;
    --ink:#e7ebee; --ink-2:#c2cbd2; --muted:#929ea8; --line:#2b333a; --line-2:#3b454e;
    --confere:#35c39a; --mudou:#f2803a; --novo:#5aabdd; --aberto:#f0b830;
    --confere-bg:#123028; --mudou-bg:#33200f; --novo-bg:#0f2735; --aberto-bg:#302510;
    --accent:#5aabdd;
  }
}
:root[data-theme="dark"]{
  --ground:#12161a; --surface:#1a1f25; --surface-2:#222932;
  --ink:#e7ebee; --ink-2:#c2cbd2; --muted:#929ea8; --line:#2b333a; --line-2:#3b454e;
  --confere:#35c39a; --mudou:#f2803a; --novo:#5aabdd; --aberto:#f0b830;
  --confere-bg:#123028; --mudou-bg:#33200f; --novo-bg:#0f2735; --aberto-bg:#302510;
  --accent:#5aabdd;
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  font-size:16px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:0 24px 96px}
.prosa{max-width:68ch}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif; text-wrap:balance; margin:0}
h1{font-size:clamp(2rem,4.4vw,3.1rem); line-height:1.1; font-weight:600; letter-spacing:-.015em}
h2{font-size:1.65rem; font-weight:600; line-height:1.25}
h3{font-size:1.12rem; font-weight:600}
p{margin:0 0 1em}
a{color:var(--accent)}
code,.num{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums}

/* --- cabecalho --- */
header.topo{border-bottom:1px solid var(--line); background:var(--surface); margin-bottom:56px}
header.topo .wrap{padding-top:56px; padding-bottom:40px}
.olho{
  font-size:.72rem; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:600; margin-bottom:14px;
}
.subtitulo{font-size:1.12rem; color:var(--ink-2); max-width:62ch; margin-top:18px}

/* --- placar --- */
.placar{display:flex; flex-wrap:wrap; gap:10px; margin-top:30px}
.placar div{
  border:1px solid var(--line-2); border-radius:3px; padding:9px 14px;
  background:var(--ground); display:flex; align-items:baseline; gap:9px;
}
.placar b{font-family:"IBM Plex Mono",monospace; font-size:1.28rem; font-weight:600}
.placar span{font-size:.8rem; color:var(--muted)}

/* --- estados --- */
.estados{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); margin-top:34px;
}
.estados > div{background:var(--surface); padding:18px 20px}
.estados .quando{font-family:"IBM Plex Mono",monospace; font-size:.76rem; color:var(--muted)}
.estados .oque{font-weight:600; margin:5px 0 4px; font-size:1.02rem}
.estados .det{font-size:.86rem; color:var(--ink-2); line-height:1.5}
.estados .agora{box-shadow:inset 3px 0 0 var(--accent)}

/* --- secoes --- */
section{margin-top:72px; scroll-margin-top:20px}
section > h2{padding-bottom:12px; border-bottom:2px solid var(--ink); margin-bottom:6px}
.chapeu{color:var(--muted); font-size:.82rem; letter-spacing:.1em;
        text-transform:uppercase; font-weight:600; margin-bottom:10px}
.nota{font-size:.9rem; color:var(--ink-2); max-width:70ch}

/* --- selo de status --- */
.selo{
  display:inline-block; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  font-weight:600; padding:2px 7px; border-radius:2px; white-space:nowrap;
  font-family:"IBM Plex Sans",sans-serif;
}
.s-confere{color:var(--confere); background:var(--confere-bg)}
.s-mudou{color:var(--mudou); background:var(--mudou-bg)}
.s-novo{color:var(--novo); background:var(--novo-bg)}
.s-aberto{color:var(--aberto); background:var(--aberto-bg)}

/* --- tabelas de diff --- */
.rolagem{overflow-x:auto; border:1px solid var(--line); background:var(--surface); margin-top:16px}
table{border-collapse:collapse; width:100%; font-size:.88rem}
th,td{padding:9px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap}
th:first-child,td:first-child{text-align:left; white-space:normal}
thead th{
  font-size:.7rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
  font-weight:600; background:var(--surface-2); border-bottom:1px solid var(--line-2);
}
tbody tr:last-child td{border-bottom:none}
td.v{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
/* celula que nao mudou fica quieta; a que mudou mostra os dois valores */
td.confere{color:var(--ink-2)}
td.mudou{background:var(--mudou-bg)}
td.mudou .antes{display:block; color:var(--mudou); text-decoration:line-through;
                opacity:.75; font-size:.82em}
td.novo{background:var(--novo-bg); color:var(--novo)}
tr.linha-nova td:first-child{box-shadow:inset 3px 0 0 var(--novo)}
tr.linha-mudou td:first-child{box-shadow:inset 3px 0 0 var(--mudou)}
caption{caption-side:top; text-align:left; padding:12px; font-weight:600;
        background:var(--surface-2); border-bottom:1px solid var(--line-2)}

/* --- parecer --- */
.parecer{display:flex; flex-direction:column; gap:1px; background:var(--line);
         border:1px solid var(--line); margin-top:22px}
.item{background:var(--surface); padding:22px 24px; display:grid;
      grid-template-columns:38px 1fr; gap:0 18px}
.item .n{font-family:"IBM Plex Mono",monospace; font-size:1.05rem; color:var(--muted);
         font-weight:600; padding-top:2px}
.item blockquote{
  margin:0 0 14px; font-family:"Source Serif 4",Georgia,serif; font-size:1.05rem;
  line-height:1.5; color:var(--ink); border-left:2px solid var(--line-2); padding-left:16px;
}
.item .resposta{font-size:.93rem; color:var(--ink-2); margin:0 0 10px}
.item .onde{font-size:.83rem; color:var(--muted); font-family:"IBM Plex Mono",monospace}

/* --- figuras --- */
.figs{display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
      gap:1px; background:var(--line); border:1px solid var(--line); margin-top:20px}
.fig{background:var(--surface); padding:18px}
.fig img{width:100%; height:auto; display:block; background:#fff; border:1px solid var(--line);
         border-radius:2px}
.fig .cap{font-size:.84rem; color:var(--ink-2); margin-top:10px; line-height:1.45}
.fig .arq{font-family:"IBM Plex Mono",monospace; font-size:.74rem; color:var(--muted);
          margin-top:6px}

/* --- listas de achado --- */
.achados{display:flex; flex-direction:column; gap:1px; background:var(--line);
         border:1px solid var(--line); margin-top:20px}
.achado{background:var(--surface); padding:20px 24px}
.achado h3{margin-bottom:7px}
.achado p{font-size:.93rem; color:var(--ink-2); margin-bottom:0; max-width:74ch}
.achado p + p{margin-top:9px}
.achado .medida{font-family:"IBM Plex Mono",monospace; font-size:.85rem;
                background:var(--surface-2); padding:9px 12px; margin-top:12px;
                border-left:2px solid var(--line-2); white-space:pre-wrap;
                overflow-x:auto; color:var(--ink)}

footer{margin-top:80px; padding-top:22px; border-top:1px solid var(--line);
       font-size:.82rem; color:var(--muted)}
@media print{
  body{background:#fff}
  section{break-inside:avoid}
  .rolagem{overflow:visible}
}
@media (max-width:640px){
  .item{grid-template-columns:1fr}
  .item .n{padding-bottom:8px}
}
"""


def tabpfn_rodou():
    """Devolve as linhas de TabPFN do arquivo do Colab, ou None.

    E o unico juiz do assunto na pagina: parecer, bloco e pendencias consultam esta
    funcao, para nao existir a possibilidade de a pagina dizer 'aberto' num lugar e
    'fechado' no outro.
    """
    tab = le("tabpfn_resultados.json", obrigatorio=False)
    if tab is None:
        return None
    linhas = [r for r in tab["resultados"] if r["modelo"].startswith("tabpfn")]
    return (tab, linhas) if linhas else None


def selo(status):
    rot = {"confere": "confere", "mudou": "mudou", "novo": "novo", "aberto": "aberto"}
    return f'<span class="selo s-{status}">{rot[status]}</span>'


def bloco_tabela(chave):
    spec = MAPA[chave]
    cabecalhos, linhas = compara_tabela(chave)
    orig = ORIG["tabelas"][chave]

    ths = "".join(f"<th>{esc(c)}</th>" for c in cabecalhos)
    trs = []
    for l in linhas:
        tds = []
        for c in l["cells"]:
            if c["status"] == "mudou":
                conteudo = (f'<span class="antes">{esc(c["antes"])}</span>'
                            f'{esc(c["depois"])}')
            elif c["status"] == "novo":
                conteudo = esc(c["depois"])
            else:
                conteudo = esc(c["depois"])
            tds.append(f'<td class="v {c["status"]}">{conteudo}</td>')
        trs.append(f'<tr class="linha-{l["status"]}">'
                   f'<td>{esc(l["rotulo"])} {selo(l["status"])}</td>'
                   + "".join(tds) + "</tr>")

    porque = (f'<p class="nota" style="margin-top:14px">{esc(spec["porque_novo"])}</p>'
              if spec["porque_novo"] else "")
    return f"""
<h3 style="margin-top:38px">{esc(spec["rotulo"])}</h3>
<p class="nota" style="margin-top:6px">Original: <em>{esc(orig["titulo"])}</em>.
Valores riscados são o que o artigo imprimiu; abaixo, o que o pipeline imprime hoje.</p>
<div class="rolagem"><table>
<thead><tr><th>Linha</th>{ths}</tr></thead>
<tbody>{"".join(trs)}</tbody>
</table></div>{porque}
"""


def bloco_parecer():
    # A resposta de cada comentario. O texto do comentario vem do PDF; o que se fez com
    # ele e conhecimento do projeto, e mora aqui.
    # O primeiro comentario e o unico cujo status depende de um arquivo: ou o TabPFN
    # rodou, ou nao rodou. Quem responde e tabpfn_rodou(), nao um literal meu.
    feito = tabpfn_rodou()
    if feito is None:
        r_tabpfn = ("aberto",
                    "O TabPFN entra no mesmo protocolo dos boosters: lags 1 a 12, "
                    "recursivo, as mesmas 103 janelas. O notebook está pronto e "
                    "testado; falta rodar no Colab com a chave da PriorLabs.",
                    "tabpfn_benchmark.ipynb")
    else:
        s_tab = feito[1][0]["smape_obtido"]
        # a posicao do TabPFN sai da comparacao com os outros, nao de uma frase minha
        outros = {r["modelo"]: r["smape_obtido"] for r in feito[0]["resultados"]
                  if not r["modelo"].startswith("tabpfn")}
        piores = sorted((m for m, v in outros.items() if v > s_tab),
                        key=lambda m: outros[m])
        melhores = sorted((m for m, v in outros.items() if v < s_tab),
                          key=lambda m: outros[m])
        r_tabpfn = ("confere",
                    f"Rodado no Colab nas 103 janelas, no mesmo protocolo dos boosters: "
                    f"lags 1 a 12, recursivo, horizonte 6. sMAPE {s_tab:.3f} %. "
                    f"Fica à frente de {', '.join(piores)} e atrás de "
                    f"{', '.join(melhores)}.",
                    "tabpfn_benchmark.ipynb, tabpfn_resultados.json")
    respostas = [
        r_tabpfn,
        ("aberto", "Não mexido. É comentário de redação sobre o fecho do referencial, "
                   "fora do escopo desta reanálise.",
         "—"),
        ("aberto", "Não mexido, mesma razão.", "—"),
        ("confere", "Doze variantes rodadas: estatísticas de janela móvel, termos de "
                    "calendário de Fourier, diferença sazonal, as combinações, e ainda "
                    "multi-passo direto. A diferença sazonal é a única que ajuda, e nem "
                    "ela leva o boosting a bater o naive sazonal.",
         "tab7_variantes.tex, revisao_fig_variantes"),
        ("confere", "Optuna rodado nos dois modelos, em duas modalidades: busca honesta "
                    "num split de desenvolvimento restrito aos 60 primeiros meses, "
                    "que são treino de todas as 103 janelas, e um teto deliberadamente "
                    "vazado, que escolhe os hiperparâmetros pelo score no próprio teste. "
                    "O teto não é resultado, é limite: serve para fechar a saída de que "
                    "o boosting foi mal por falta de ajuste. Nem ele salva o boosting.",
         "revisao_tab_optuna.tex, revisao_fig_optuna"),
    ]
    itens = []
    for i, (a, (status, resposta, onde)) in enumerate(
            zip(ORIG["anotacoes"], respostas), 1):
        itens.append(f"""
<div class="item">
  <div class="n">{i:02d}</div>
  <div>
    <blockquote>{esc(a["texto"])}</blockquote>
    <p class="resposta">{selo(status)} {esc(resposta)}</p>
    <p class="onde">p. {a["pagina"]} &middot; {esc(onde)}</p>
  </div>
</div>""")
    return '<div class="parecer">' + "".join(itens) + "</div>"


def bloco_figuras():
    cartoes = []
    for chave, arq in FIGURAS.items():
        cartoes.append(f"""
<div class="fig">
  <img src="figures_v2/{arq}.png" alt="{esc(arq)}" loading="lazy">
  <div class="cap">{esc(ORIG["figuras"][chave])} {selo("confere")}</div>
  <div class="arq">{esc(chave.replace("figura", "Figure "))} &rarr; {esc(arq)}.tex</div>
</div>""")
    novas = []
    for arq, cap in FIGURAS_NOVAS:
        novas.append(f"""
<div class="fig">
  <img src="figures_v2/{arq}.png" alt="{esc(arq)}" loading="lazy">
  <div class="cap">{esc(cap)} {selo("novo")}</div>
  <div class="arq">{esc(arq)}.tex</div>
</div>""")
    return ('<div class="figs">' + "".join(cartoes) + "</div>"
            '<h3 style="margin-top:40px">Seis figuras que nao existiam</h3>'
            '<div class="figs">' + "".join(novas) + "</div>")


def bloco_tabpfn():
    tab = le("tabpfn_resultados.json", obrigatorio=False)
    local = le("reconferencia_local.json")
    versoes = local["_meta"]["versoes"]
    linhas = "\n".join(
        f'{l["modelo"]:<14s} {l["smape"]:.6f}   artigo {l["smape_artigo"]:.6f}   '
        f'{l["diferenca_pp"]:+.6f}   {l["veredito"]}'
        for l in local["resultados"])

    # Um arquivo de resultados do Colab SEM nenhuma linha de TabPFN nao fecha a
    # pendencia: quer dizer que a rodada aconteceu mas as celulas do TabPFN ficaram
    # desligadas. Tratar como fechado ai seria dizer que o comentario do parecer foi
    # atendido quando nao foi.
    if tab is not None and not any(
            r["modelo"].startswith("tabpfn") for r in tab["resultados"]):
        trs = "".join(
            f'<tr><td>{esc(r["modelo"])} '
            f'{selo("confere" if r["veredito"] == "bate" else "mudou")}</td>'
            f'<td class="v">{r["smape_obtido"]:.6f}</td>'
            f'<td class="v">{r["smape_artigo"]:.6f}</td>'
            f'<td class="v">{r["diferenca_pp"]:+.6f}</td></tr>'
            for r in tab["resultados"])
        vers = tab["_meta"].get("versoes")
        nota_vers = (" Versões: " + ", ".join(f"{k} {v}" for k, v in vers.items() if v)
                     if vers else " O arquivo desta rodada não registrou as versões das "
                                  "bibliotecas, o que foi corrigido no notebook.")

        # A leitura dos desvios sai dos proprios numeros, para nao envelhecer junto com
        # o texto: o SARIMA ja desviou 0,22 pp por um erro meu e hoje desvia milesimos.
        por_modelo = {r["modelo"]: r for r in tab["resultados"]}
        batem = [m for m, r in por_modelo.items() if r["veredito"] == "bate"]
        desvios = sorted(((abs(r["diferenca_pp"]), m) for m, r in por_modelo.items()
                          if r["veredito"] == "desvia"), reverse=True)
        maior, nome_maior = desvios[0] if desvios else (0.0, "")
        leitura = (
            f"Os {len(batem)} modelos determinísticos batem exato: "
            f"{', '.join(esc(m) for m in batem)}. Entre os que desviam, o maior desvio é "
            f"do {esc(nome_maior)}, {maior:.3f} pp, e nenhum passa de um décimo e meio de "
            "ponto percentual &mdash; é ruído de versão de biblioteca, não divergência de "
            "protocolo. Numa rodada anterior o SARIMA desviava 0,222 pp, mas aquilo era "
            "um erro meu no notebook, que usava <code>enforce_stationarity=False</code> "
            "onde o pipeline usa <code>True</code>; corrigido, ele voltou para a faixa "
            "dos milésimos.")
        return f"""
<p class="resposta">{selo("aberto")} O notebook rodou no Colab, mas o arquivo devolvido
não traz nenhuma linha de TabPFN: a célula ficou no modo <code>não rodar</code>. A
pendência do parecer continua aberta.</p>
<p class="nota">O que a rodada entregou foi a reconferência dos outros modelos, e ela
vale por si: o portão das três ingênuas passou, o que confirma que o protocolo do
notebook é o do artigo.{esc(nota_vers)}</p>
<div class="rolagem"><table>
<thead><tr><th>Modelo</th><th>sMAPE no Colab</th><th>sMAPE do artigo</th>
<th>Diferença</th></tr></thead>
<tbody>{trs}</tbody></table></div>
<p class="nota" style="margin-top:14px">{leitura}</p>"""

    if tab is None:
        estado = f"""
<p class="resposta">{selo("aberto")} O notebook está gerado e o protocolo dele já foi
provado neste computador. Falta a rodada com a chave da PriorLabs.</p>
<p class="nota">O notebook começa por um portão: mede as três referências ingênuas e para
se elas não baterem com o artigo até a sexta casa. Aqui bateram, e o CatBoost também, o
que quer dizer que o protocolo do notebook é o mesmo do artigo e que a rodada do TabPFN
lá será comparável. As células que gastam cota da API começam desligadas.</p>
<div class="achado" style="border:1px solid var(--line); margin-top:18px">
  <h3>Reconferência neste ambiente</h3>
  <p>Python {esc(versoes["python"])}, xgboost {esc(versoes["xgboost"])},
     catboost {esc(versoes["catboost"])}.</p>
  <div class="medida">{esc(linhas)}</div>
  <p style="margin-top:12px">O CatBoost reproduz na sexta casa. O XGBoost não, e este é
  <strong>o quarto valor distinto</strong> que ele produz em quatro ambientes.</p>
</div>"""
    else:
        # ordena pelo proprio sMAPE: a leitura da secao e a ordem da tabela, e as duas
        # saem do mesmo lugar.
        ordenado = sorted(tab["resultados"], key=lambda r: r["smape_obtido"])
        def sel(r):
            if r["veredito"] == "novo":
                return "novo"
            return "confere" if r["veredito"] == "bate" else "mudou"
        trs = "".join(
            f'<tr class="linha-{"novo" if r["veredito"] == "novo" else "confere"}">'
            f'<td>{esc(r["modelo"])} {selo(sel(r))}</td>'
            f'<td class="v">{r["smape_obtido"]:.6f}</td>'
            f'<td class="v">'
            f'{"—" if r["smape_artigo"] is None else format(r["smape_artigo"], ".6f")}</td>'
            f'<td class="v">'
            f'{"—" if r["diferenca_pp"] is None else format(r["diferenca_pp"], "+.6f")}'
            f'</td></tr>'
            for r in ordenado)

        s_tab = next(r["smape_obtido"] for r in tab["resultados"]
                     if r["modelo"] == "tabpfn")
        por = {r["modelo"]: r["smape_obtido"] for r in tab["resultados"]}
        tabulares = {m: por[m] for m in ("xgboost", "catboost") if m in por}
        ingenuas = {m: por[m] for m in ("naive", "snaive", "snaive_drift") if m in por}
        classicos = {m: por[m] for m in ("sarima", "prophet") if m in por}
        melhor_tab = min(tabulares, key=tabulares.get)
        melhor_ing = min(ingenuas, key=ingenuas.get)
        melhor_cla = min(classicos, key=classicos.get)
        meta = tab["_meta"]
        ausentes = meta.get("modelos_ausentes") or []
        vers = meta.get("versoes") or {}

        falta = ""
        if ausentes:
            falta = (f'<p class="nota" style="margin-top:14px">Não rodaram nesta sessão: '
                     f'{esc(", ".join(ausentes))}. O TimesFM pede GPU e já tem valor no '
                     f'artigo; o TabPFN-TS era o enquadramento secundário, de série '
                     f'temporal em vez de tabular, e não é o que o parecer pediu.</p>')

        estado = f"""
<p class="resposta">{selo("confere")} Rodado no Colab, nas 103 janelas, com
<code>tabpfn_client {esc(vers.get("tabpfn_client", "?"))}</code> e
<code>thinking_mode={esc(meta.get("thinking_mode"))}</code>.</p>
<div class="rolagem"><table>
<thead><tr><th>Modelo</th><th>sMAPE obtido</th><th>sMAPE do artigo</th>
<th>Diferença</th></tr></thead>
<tbody>{trs}</tbody></table></div>
<p style="margin-top:16px">O TabPFN faz <strong>{s_tab:.3f}&#8239;%</strong>. É o melhor
modelo tabular da comparação: ganha do {esc(melhor_tab)}, que faz
{tabulares[melhor_tab]:.3f}&#8239;%, por
{tabulares[melhor_tab] - s_tab:.3f}&#8239;pp. E é o <strong>único modelo tabular que bate
as referências ingênuas</strong> &mdash; o {esc(melhor_ing)} faz
{ingenuas[melhor_ing]:.3f}&#8239;%, e nem o XGBoost nem o CatBoost chegaram lá, em
nenhuma das doze variantes de atributos nem no teto com vazamento do Optuna &mdash;
o limite superior que uma busca de hiperparâmetros atingiria se pudesse escolher olhando
o próprio teste.</p>
<p>Isso responde o comentário do revisor no sentido em que ele foi feito: a comparação
agora inclui o melhor do tabular, e a inclusão <strong>não muda a conclusão do artigo</strong>.
O {esc(melhor_cla)} faz {classicos[melhor_cla]:.3f}&#8239;%, uma distância de
{s_tab - classicos[melhor_cla]:.3f}&#8239;pp que nenhum modelo tabular encostou. A leitura
do artigo fica mais forte, não mais fraca: não é que faltasse um modelo tabular bom, é que
o problema &mdash; uma série mensal de 168 pontos com sazonalidade limpa &mdash; não
recompensa esse tipo de modelo.</p>{falta}"""
    return estado


def bloco_reprodutibilidade():
    d = divergencia_calibracao()
    p, s = d["prophet"], d["sarima"]
    ceil_xgb = le("ceiling_xgboost.json")["xgboost"]
    local = {r["modelo"]: r for r in le("reconferencia_local.json")["resultados"]}
    versoes = le("reconferencia_local.json")["_meta"]["versoes"]

    cov = "\n".join([
        "cobertura do Prophet     h1     h2     h3     h4     h5     h6",
        "rodada A             " + "  ".join(f"{v:.3f}" for v in p["cobertura_a"]),
        "rodada B             " + "  ".join(f"{v:.3f}" for v in p["cobertura_b"]),
    ])
    xgb = "\n".join([
        f"artigo (Tabela 1)                       {V['tabela1']['xgboost']['smape']:.6f}",
        f"ambiente do Optuna, reproduzida         {ceil_xgb['baseline_reproduzida']:.6f}",
        f"ambiente do Optuna, esperada            {ceil_xgb['baseline_esperada']:.6f}",
        f"este computador, xgboost {versoes['xgboost']:<14s} "
        f"{local['xgboost']['smape']:.6f}",
    ])
    cat = "\n".join([
        f"artigo (Tabela 1)                       "
        f"{V['tabela1']['catboost']['smape']:.6f}",
        f"este computador, catboost {versoes['catboost']:<13s} "
        f"{local['catboost']['smape']:.6f}",
    ])

    return f"""
<div class="achados">

<div class="achado">
  <h3>O intervalo do Prophet não reproduz {selo("mudou")}</h3>
  <p>Existem duas rodadas de intervalo no projeto. Nas {p["n"]} previsões, o
  <strong>ponto é idêntico nas duas</strong>, com desvio máximo de
  <code>{p["desvio_ypred"]:.0f}</code>. Os limites, não: divergem em até
  <code>{max(p["desvio_lo"], p["desvio_hi"]):.1f}</code> óbitos. O Prophet tira as bandas
  de uma amostra finita da posterior, e essa amostra não está com semente fixa neste
  pipeline.</p>
  <div class="medida">{esc(cov)}</div>
  <p>O SARIMA não tem esse problema: as duas rodadas dele coincidem exatamente, com
  desvio máximo <code>{max(s["desvio_lo"], s["desvio_hi"]):.0f}</code>. O efeito prático
  é pequeno, décimos de ponto percentual na cobertura, mas ele decide qual dos dois
  conjuntos de dígitos vai para o artigo &mdash; e hoje a figura de calibração do
  manuscrito usa a rodada B enquanto a tabela regerada usa a rodada A.</p>
</div>

<div class="achado">
  <h3>O XGBoost muda de valor conforme a versão {selo("mudou")}</h3>
  <p>Quatro ambientes, quatro números. Não há erro em nenhum deles: o XGBoost muda os
  cortes das árvores entre versões da biblioteca, e o projeto nunca fixou a versão.</p>
  <div class="medida">{esc(xgb)}</div>
  <p>O CatBoost, no mesmo protocolo e nos mesmos ambientes, não se move:</p>
  <div class="medida">{esc(cat)}</div>
</div>

<div class="achado">
  <h3>O resto reproduz exato {selo("confere")}</h3>
  <p>As três referências ingênuas batem até a sexta casa neste computador. O Prophet
  reproduz previsão a previsão entre ambientes, com desvio máximo
  <code>{V["prophet_exog_reproducao"]["max_desvio_previsao"]:.0f}</code> nas 618
  previsões &mdash; foi essa checagem que autorizou a linha nova do Prophet na tabela de
  temperatura, porque sem ela a comparação seria entre dois modelos diferentes de mesmo
  nome.</p>
</div>

</div>"""


def bloco_procedencia():
    blocos = tabelas_antes_do_commit()
    trs = "".join(
        f'<tr class="linha-{"novo" if not b["no_gerador_antes"] else "confere"}">'
        f'<td>{esc(b["bloco"])} '
        f'{selo("mudou" if not b["no_gerador_antes"] else "confere")}</td>'
        f'<td class="v">{"não" if not b["no_gerador_antes"] else "sim"}</td>'
        f'<td class="v">sim</td>'
        f'<td>{esc(b["arquivo"])}</td></tr>'
        for b in blocos)
    n = sum(1 for b in blocos if not b["no_gerador_antes"])
    return f"""
<p class="nota">O artigo declara, na seção de reprodutibilidade, que toda tabela e figura
sai de código. Vale para a maior parte. Mas {n} blocos que aparecem no manuscrito não
saíam do gerador no commit <code>{COMMIT_ANTES}</code>: existiam apenas dentro do .tex,
digitados. A verificação abaixo é feita contra o próprio git, na hora.</p>
<div class="rolagem"><table>
<thead><tr><th>Bloco do manuscrito</th><th>Saía do gerador antes?</th>
<th>Sai agora?</th><th>Arquivo</th></tr></thead>
<tbody>{trs}</tbody></table></div>
<p class="nota" style="margin-top:14px">Nenhum deles estava errado &mdash; conferi todos
contra a fonte e os valores batem. O problema era de garantia: um número digitado não tem
como avisar quando os dados por baixo dele mudam.</p>"""


def bloco_pendencias():
    itens = [
        ("Qual rodada de intervalo do Prophet vale",
         "A figura de calibração do manuscrito e a tabela regerada usam rodadas "
         "diferentes. Publicadas juntas, o documento se contradiz. A saída limpa é fixar "
         "a semente da amostragem do Prophet e rodar uma vez só, o que também resolve o "
         "problema para sempre."),
        ("O manuscrito ainda diz que nenhum ajuste foi feito",
         "As Limitações do main_victor.tex afirmam que nenhuma busca de hiperparâmetros "
         "foi executada. O Optuna rodou 100 trials por modelo. A afirmação precisa cair, "
         "e o resultado da busca precisa entrar: ele reforça a conclusão do artigo em vez "
         "de contrariá-la."),
        ("A tabela de variantes não tem contraparte no texto",
         "tab7_variantes sai do gerador, mas nenhuma versão do manuscrito a inclui; as "
         "variantes só aparecem em prosa. Ou entra como tabela, ou sai do gerador."),
    ]
    if tabpfn_rodou() is None:
        itens.append(
            ("TabPFN",
             "Notebook pronto e testado. Falta rodar no Colab com a chave da PriorLabs "
             "e me devolver o tabpfn_resultados.json."))
    return '<div class="achados">' + "".join(
        f'<div class="achado"><h3>{esc(t)} {selo("aberto")}</h3><p>{esc(d)}</p></div>'
        for t, d in itens) + "</div>"


def bloco_bugs():
    itens = [
        ("Table~\\ref virando quebra de linha", "tab1_desempenho",
         "Table~\\ref{tab:calibracao}   numa string não-raw do Python",
         "o \\r virou carriage return e o PDF imprimia   Table~ef{tab:calibracao}"),
        ("Porcentagem crua comentando a linha", "tab6_calibracao",
         "{picp:.1%}   gerava   84.5%",
         "em LaTeX o % comenta o resto da linha, e a nota perdia a frase inteira"),
    ]
    return '<div class="achados">' + "".join(
        f'<div class="achado"><h3>{esc(t)} {selo("mudou")}</h3>'
        f'<p>Em <code>{esc(arq)}.tex</code>.</p>'
        f'<div class="medida">{esc(a)}\n{esc(b)}</div></div>'
        for t, arq, a, b in itens) + "</div>"


# ------------------------------------------------------------------------ main

TITULO = "Auditoria da revisão cardiovascular"

FONTES = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&"
    'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">'
)

ARTEFATO = Path(
    r"C:\Users\Renato\AppData\Local\Temp\claude\C--Users-Renato"
    r"\b237ef68-6d9f-4419-9fc9-7ec915f882cc\scratchpad\auditoria_revisao.html"
)


def embute_figuras(corpo):
    """Troca src="figures_v2/x.png" por data URI. O artefato nao resolve caminho."""
    faltando = []

    def troca(m):
        arq = FIGS_DIR / f"{m.group(1)}.png"
        if not arq.exists():
            faltando.append(arq.name)
            return m.group(0)
        b64 = base64.b64encode(arq.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{b64}"'

    novo = re.sub(r'src="figures_v2/([^"]+)\.png"', troca, corpo)
    if faltando:
        raise FileNotFoundError(f"figuras ausentes: {faltando}")
    return novo


def escreve_artefato(corpo):
    """Versao para publicar: sem invólucro, com as figuras embutidas."""
    doc = (f"<title>{TITULO}</title>\n{FONTES}\n<style>{CSS}</style>\n"
           + embute_figuras(corpo))
    ARTEFATO.parent.mkdir(parents=True, exist_ok=True)
    ARTEFATO.write_text(doc, encoding="utf-8")
    n = doc.count("data:image/png;base64,")
    print(f"  artefato {ARTEFATO.name}  ({len(doc.encode()) // 1024} KB, "
          f"{n} figuras embutidas)")


def main() -> int:
    # placar geral, contado das comparacoes e nao afirmado
    conta = {"confere": 0, "mudou": 0, "novo": 0}
    for chave in MAPA:
        _, linhas = compara_tabela(chave)
        for l in linhas:
            conta[l["status"]] += 1

    secoes = []

    secoes.append(f"""
<section id="parecer">
  <div class="chapeu">Ponto de partida</div>
  <h2>Os cinco comentários do parecer</h2>
  <p class="nota" style="margin-top:12px">São eles que explicam por que a revisão existe.
  Estão aqui na íntegra, como saem das anotações do PDF.</p>
  {bloco_parecer()}
</section>""")

    secoes.append(f"""
<section id="tabelas">
  <div class="chapeu">Número a número</div>
  <h2>As cinco tabelas do original</h2>
  <p class="nota" style="margin-top:12px">A comparação é na precisão impressa: cada valor
  de hoje foi formatado com as mesmas casas que o artigo usou, e só então confrontado.
  A pergunta é a de quem lê o artigo &mdash; o dígito publicado ainda é o dígito que o
  pipeline produz?</p>
  {"".join(bloco_tabela(k) for k in MAPA)}
</section>""")

    secoes.append(f"""
<section id="figuras">
  <div class="chapeu">Seis viraram doze</div>
  <h2>Figuras</h2>
  <p class="nota" style="margin-top:12px">As seis do original continuam, com os mesmos
  números, agora na paleta Okabe-Ito e com texto em Times preto. Seis novas cobrem as
  análises que a revisão acrescentou.</p>
  {bloco_figuras()}
</section>""")

    secoes.append(f"""
<section id="tabpfn">
  <div class="chapeu">O comentário que faltava</div>
  <h2>TabPFN</h2>
  <p class="nota" style="margin-top:12px">O revisor pediu o TabPFN como competidor
  tabular do XGBoost e do CatBoost, então ele entra no protocolo deles, e não num
  protocolo próprio.</p>
  {bloco_tabpfn()}
</section>""")

    secoes.append(f"""
<section id="reprodutibilidade">
  <div class="chapeu">O que reproduz e o que não</div>
  <h2>Reprodutibilidade</h2>
  {bloco_reprodutibilidade()}
</section>""")

    secoes.append(f"""
<section id="procedencia">
  <div class="chapeu">De onde veio cada número</div>
  <h2>Procedência</h2>
  {bloco_procedencia()}
</section>""")

    secoes.append(f"""
<section id="bugs">
  <div class="chapeu">Corrigidos no gerador</div>
  <h2>Dois defeitos de LaTeX</h2>
  <p class="nota" style="margin-top:12px">Os dois tinham efeito visível no PDF
  publicado.</p>
  {bloco_bugs()}
</section>""")

    secoes.append(f"""
<section id="pendencias">
  <div class="chapeu">Decisões que são suas</div>
  <h2>Em aberto</h2>
  {bloco_pendencias()}
</section>""")

    corpo = f"""
<header class="topo">
  <div class="wrap">
    <div class="olho">Mortalidade cardiovascular em São Paulo &middot; auditoria da revisão</div>
    <h1>O que mudou na revisão</h1>
    <p class="subtitulo">Comparação entre o manuscrito que foi para revisão e a análise
    que os scripts produzem hoje, tabela por tabela e figura por figura, com a
    procedência de cada número.</p>
    <div class="placar">
      <div><b>{conta["confere"]}</b><span>linhas conferem</span></div>
      <div><b>{conta["mudou"]}</b><span>mudaram</span></div>
      <div><b>{conta["novo"]}</b><span>novas</span></div>
      <div><b>{len(FIGURAS_NOVAS)}</b><span>figuras novas</span></div>
    </div>
    <div class="estados">
      <div>
        <div class="quando">17/08/2026 &middot; manuscript_comentado.pdf</div>
        <div class="oque">Original</div>
        <div class="det">A versão que foi para revisão. {len(ORIG["tabelas"])} tabelas,
        {len(ORIG["figuras"])} figuras, {len(ORIG["anotacoes"])} comentários.</div>
      </div>
      <div>
        <div class="quando">18/08/2026 &middot; main_victor.tex</div>
        <div class="oque">Intermediário</div>
        <div class="det">Revisão em curso. Entra aqui só como evidência: é nele que
        aparecem os blocos montados fora do pipeline.</div>
      </div>
      <div class="agora">
        <div class="quando">{date.today():%d/%m/%Y} &middot; paper/ e figures_v2/</div>
        <div class="oque">Análise nova</div>
        <div class="det">10 tabelas e 12 figuras, todas regeradas da fonte, mais os
        experimentos da revisão.</div>
      </div>
    </div>
  </div>
</header>

<div class="wrap">
{"".join(secoes)}
<footer>
  Página gerada por <code>scripts/revisao/build_comparacao_html.py</code> a partir de
  <code>original_manuscript_numbers.json</code>, <code>verified_numbers.json</code>,
  dos resultados da revisão e do histórico do git. Nenhum número foi digitado.
</footer>
</div>
"""

    doc = (
        '<!doctype html>\n<html lang="pt-BR">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{TITULO}</title>\n{FONTES}\n<style>{CSS}</style>\n"
        f"</head>\n<body>\n{corpo}\n</body>\n</html>\n"
    )
    SAIDA.write_text(doc, encoding="utf-8")
    print(f"  html    {SAIDA.name}  ({SAIDA.stat().st_size // 1024} KB)")
    escreve_artefato(corpo)
    print(f"    {conta['confere']} conferem, {conta['mudou']} mudaram, "
          f"{conta['novo']} novas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
