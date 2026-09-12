#!/usr/bin/env python3
"""Extrai do manuscrito comentado (a versao que foi para revisao) os numeros, as
anotacoes do revisor e os trechos de prosa que a comparacao cita.

Este e o unico script do projeto que le numero de PDF. Existe porque o manuscrito
comentado e a unica forma em que a versao original sobreviveu: nao ha .tex dela, e o
commit correspondente do repositorio ja traz parte da revisao. Sem esta extracao a
comparacao "original x nova" teria de ser digitada a mao, que e exatamente o que o
resto do pipeline nao faz.

O texto do PDF sai uma celula por linha em algumas tabelas e varias celulas na mesma
linha em outras, entao a leitura e por TOKEN e nao por linha: dentro do bloco de cada
tabela, depois do rotulo da linha, tomam-se os primeiros k tokens que casem com
intervalo, "N of M" ou numero decimal. Inteiros nus ficam de fora de proposito, porque
o PDF tem numeracao de linha e de pagina solta no meio do texto e nenhuma celula das
cinco tabelas e um inteiro nu.

Saida: results/revisao/original_manuscript_numbers.json

Uso:
    python scripts/revisao/extrai_original.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import RES  # noqa: E402

PDF = Path(
    r"G:\.shortcut-targets-by-id\1zRZlDXqjBRVUlO0Zys2L0W69eJvgMI58\Labs"
    r"\Cardiovascular Time Series\IJF_Series_Temporais_CV\00_manuscrito"
    r"\manuscript_comentado.pdf"
)
SAIDA = RES / "revisao" / "original_manuscript_numbers.json"

# ordem importa: "N of M" antes de numero, senao o "0" seria capturado sozinho
TOKEN = re.compile(
    r"\[[^\]]*\]"           # intervalo, [4.17, 5.27]
    r"|\d+\s+of\s+\d+"      # celula de DM, 0 of 6
    r"|[\u2212+-]?\d+\.\d+"  # decimal, com menos tipografico ou hifen
)

# Cada tabela: rotulos de linha na ordem em que aparecem, nomes das colunas, e o
# rotulo do bloco seguinte, que fecha a busca.
TABELAS = {
    "tabela1": {
        "titulo": "Forecasting accuracy",
        "ancora": "Table 1:",
        "fim": "Table 1 reports",
        "colunas": ["mae", "rmse", "smape", "ic", "largura"],
        "linhas": ["Prophet", "SARIMA", "TimesFM", "CatBoost", "XGBoost"],
    },
    "tabela2": {
        "titulo": "Paired differences among the leading three",
        "ancora": "Table 2:",
        "fim": "pp, percentage points",
        "colunas": ["dif", "ic", "p_boot", "dm"],
        "linhas": ["Prophet \u2212SARIMA", "Prophet \u2212TimesFM", "SARIMA \u2212TimesFM"],
    },
    "tabela3": {
        "titulo": "Effect of training history",
        "ancora": "Table 3:",
        "fim": "The short-history column",
        "colunas": ["smape_curta", "smape_longa", "ganho"],
        "linhas": ["Prophet", "SARIMA", "TimesFM"],
    },
    "tabela4": {
        "titulo": "Expanding versus sliding 60-month window",
        "ancora": "Table 4:",
        "fim": "The only difference between",
        "colunas": ["expanding", "sliding60", "dif", "ic", "p", "dm"],
        "linhas": ["Prophet", "SARIMA", "TimesFM", "CatBoost", "XGBoost"],
    },
    "tabela5": {
        "titulo": "Effect of minimum temperature",
        "ancora": "Table 5:",
        "fim": "Under the climatology policy",
        "colunas": ["sem", "com", "ganho", "ic", "dm", "teto"],
        "linhas": ["SARIMA", "CatBoost", "XGBoost"],
    },
}

# Trechos de prosa que a pagina de comparacao cita literalmente. Guardados por ancora
# de inicio e de fim para nao depender de numero de linha.
PROSA = {
    "sem_intervalos": (
        "No prediction intervals, and therefore no forecast calibration.",
        "Counts rather than rates.",
    ),
    "prophet_sem_exogena": (
        "Prophet and TimesFM do not accept exogenous regressors",
        "All three gains in Table 5",
    ),
    "periodo_pandemia": (
        "Pandemic period.",
        "Raw SIM data.",
    ),
}


# O PDF vem do LaTeX com ligaturas tipograficas: "difference" sai como um unico
# glifo "ﬀ" no meio da palavra. Sem desfazer isso, qualquer ancora de texto que
# contenha ff, fi ou fl falha silenciosamente.
LIGATURAS = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
             "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


def desliga(s: str) -> str:
    for k, v in LIGATURAS.items():
        s = s.replace(k, v)
    return s


SO_INTEIRO = re.compile(r"^\d{1,3}$")


def texto_do_pdf(doc) -> str:
    """Texto corrido, sem as linhas que sao so um inteiro.

    O manuscrito foi compilado com numeracao de linha, entao o PDF tem numeros soltos
    entre os paragrafos, e o rodape acrescenta o numero da pagina. Se ficassem, apareceriam
    no meio dos trechos de prosa citados. Nenhuma celula das cinco tabelas e um inteiro nu,
    entao remove-los nao custa nada a leitura dos numeros.
    """
    linhas = desliga("\n".join(p.get_text() for p in doc)).split("\n")
    return "\n".join(l for l in linhas if not SO_INTEIRO.match(l.strip()))


# hifen de quebra de linha: "tem- perature" volta a ser "temperature". Exige minuscula
# dos dois lados, entao nao toca em "-0.095" nem em "leakage-free", que nao tem espaco.
HIFEN_QUEBRA = re.compile(r"(?<=[a-z\u00e0-\u00ff])- (?=[a-z\u00e0-\u00ff])")


def limpa(s: str) -> str:
    """Menos tipografico para hifen, hifenacao de quebra desfeita, espacos normalizados."""
    s = re.sub(r"\s+", " ", desliga(s).replace("\u2212", "-")).strip()
    return HIFEN_QUEBRA.sub("", s)


def le_tabela(txt: str, spec: dict) -> dict:
    i = txt.index(spec["ancora"])
    j = txt.index(spec["fim"], i)
    bloco = txt[i:j]
    fora = []
    linhas = {}
    for k, rot in enumerate(spec["linhas"]):
        # a busca comeca depois do rotulo anterior, para nao casar um nome que se
        # repita no cabecalho ou na legenda
        inicio = bloco.index(rot, fora[-1] if fora else 0) + len(rot)
        fora.append(inicio)
        toks = TOKEN.findall(bloco[inicio:])
        n = len(spec["colunas"])
        if len(toks) < n:
            raise ValueError(f"{spec['ancora']} linha {rot}: {len(toks)} tokens, "
                             f"esperado {n}")
        linhas[limpa(rot)] = dict(zip(spec["colunas"], [limpa(t) for t in toks[:n]]))
    return {"titulo": spec["titulo"], "colunas": spec["colunas"], "linhas": linhas}


def le_prosa(txt: str) -> dict:
    out = {}
    for chave, (ini, fim) in PROSA.items():
        i = txt.index(ini)
        j = txt.index(fim, i)
        out[chave] = limpa(txt[i:j])
    return out


def le_anotacoes(doc) -> list:
    """Os comentarios do revisor. Sao a razao de ser da revisao inteira."""
    out = []
    for i, pag in enumerate(doc, 1):
        for a in pag.annots() or []:
            info = a.info
            conteudo = (info.get("content") or "").strip()
            if not conteudo:
                continue
            out.append({
                "pagina": i,
                "tipo": a.type[1],
                "texto": limpa(conteudo),
            })
    return out


def le_figuras(txt: str) -> dict:
    """Primeira frase da legenda de cada figura, pela ancora 'Figure N:'.

    So a primeira frase: e ela que descreve o que a figura mostra, e o resto da legenda
    e comentario que a pagina de comparacao nao usa. Alem disso a legenda completa nao
    tem delimitador confiavel no texto extraido, ela emenda no corpo do paragrafo
    seguinte sem linha em branco.
    """
    out = {}
    for n in range(1, 7):
        i = txt.index(f"Figure {n}:")
        janela = limpa(txt[i:i + 500])
        m = re.match(rf"Figure {n}:\s*(.+?\.)(?:\s|$)", janela)
        if not m:
            raise ValueError(f"legenda da Figure {n} sem frase delimitada")
        out[f"figura{n}"] = m.group(1)
    return out


def main() -> int:
    doc = fitz.open(PDF)
    txt = texto_do_pdf(doc)
    dados = {
        "_meta": {
            "gerado_por": "scripts/revisao/extrai_original.py",
            "fonte": PDF.name,
            "paginas": doc.page_count,
            "regra": "numeros lidos do PDF por token, nunca digitados",
        },
        "anotacoes": le_anotacoes(doc),
        "tabelas": {k: le_tabela(txt, s) for k, s in TABELAS.items()},
        "figuras": le_figuras(txt),
        "prosa": le_prosa(txt),
    }
    SAIDA.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  json    {SAIDA.relative_to(ROOT)}")
    print(f"    {len(dados['anotacoes'])} anotacoes, "
          f"{len(dados['tabelas'])} tabelas, {len(dados['figuras'])} figuras, "
          f"{len(dados['prosa'])} trechos de prosa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
