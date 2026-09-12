#!/usr/bin/env python3
"""Roda localmente as celulas do notebook do Colab que nao precisam de rede nem de chave.

Serve a dois propositos. Primeiro, e o teste do notebook: se as tres referencias
ingenuas nao baterem com o artigo aqui, o protocolo do notebook divergiu e nao vale
gastar cota da API da PriorLabs no Colab. Segundo, a rodada e ela propria um dado: e
mais um ambiente independente medindo os mesmos modelos, e e assim que a divergencia
de versao do XGBoost fica documentada com a versao ao lado do numero.

Executa as celulas do .ipynb em vez de reimplementa-las, para que o que e testado aqui
seja literalmente o codigo que vai rodar la.

Saida: results/revisao/reconferencia_local.json

Uso:
    python scripts/revisao/reconfere_local.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import RES  # noqa: E402
from build_tabpfn_notebook import SAIDA as NOTEBOOK  # noqa: E402

SAIDA = RES / "revisao" / "reconferencia_local.json"

# Celulas do notebook que rodam sem rede e sem chave: dados, protocolo, ingenuas e
# boosting. As de SARIMA/Prophet ficam de fora porque levam minutos e ja tem
# reproducao verificada no pipeline oficial; as de TabPFN precisam da API.
CELULAS = {"dados": 3, "protocolo": 4, "ingenuas": 6, "boosting": 8}


def versoes() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__}
    for nome in ("xgboost", "catboost"):
        try:
            out[nome] = __import__(nome).__version__
        except Exception:
            out[nome] = None
    return out


def main() -> int:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celulas = nb["cells"]

    escopo = {"np": np, "pd": pd, "time": time, "json": json,
              "os": __import__("os"), "re": __import__("re"),
              "warnings": __import__("warnings")}
    for nome, i in CELULAS.items():
        # ''.join, nao '\n'.join: e assim que o Jupyter remonta a celula. Juntar com
        # '\n' aqui esconderia um notebook cujas linhas foram gravadas sem quebra --
        # que foi o defeito da primeira versao deste conjunto.
        fonte = "".join(celulas[i]["source"])
        print(f"--- celula {i} ({nome})")
        exec(fonte, escopo)

    obtidos = escopo["RESULTADOS"]
    ref = escopo["REFERENCIA"]["smape"]
    linhas = []
    for m, v in obtidos.items():
        d = v - ref[m]
        linhas.append({"modelo": m, "smape": float(v), "smape_artigo": float(ref[m]),
                       "diferenca_pp": float(d),
                       "veredito": "bate" if abs(d) < 1e-6 else "desvia"})

    dados = {
        "_meta": {
            "gerado_por": "scripts/revisao/reconfere_local.py",
            "executa": f"celulas {sorted(CELULAS.values())} de {NOTEBOOK.name}",
            "protocolo": "103 janelas, horizonte 6, minimo de treino 60, expansiva",
            "versoes": versoes(),
        },
        "resultados": linhas,
    }
    SAIDA.write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  json    {SAIDA.relative_to(ROOT)}")
    for l in linhas:
        print(f"    {l['modelo']:14s} {l['smape']:.6f}  "
              f"({l['diferenca_pp']:+.6f})  {l['veredito']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
