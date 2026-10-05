"""Montagem e validacao de notebook gerado, compartilhada pelos geradores.

Estava duplicada entre os geradores de notebook, e a duplicacao era perigosa: a parte
que importa aqui e a regra do `keepends` em `fonte`, que ja quebrou um notebook inteiro
uma vez. Dois lugares para consertar significa um lugar para esquecer.
"""
from __future__ import annotations


def fonte(txt: str) -> list[str]:
    """Quebra em linhas MANTENDO o \\n de cada uma.

    O nbformat manda que source seja o texto ja quebrado, e o Jupyter remonta com
    ''.join(source), sem separador. Se as linhas vierem sem o \\n final, a celula
    inteira colapsa numa linha so e o notebook nao roda. Foi exatamente esse o defeito
    da primeira versao do gerador do TabPFN.
    """
    return txt.strip("\n").splitlines(keepends=True)


def md(txt: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": fonte(txt)}


def code(txt: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": fonte(txt)}


def valida(nb: dict) -> None:
    """Remonta cada celula COMO O JUPYTER remonta e compila o Python resultante.

    A checagem tem de partir de ''.join(source), nao de '\\n'.join(source): foi por
    testar a segunda forma que a primeira versao do gerador passou num notebook em que
    toda celula colapsava numa linha unica no Colab.
    """
    for i, c in enumerate(nb["cells"]):
        texto = "".join(c["source"])
        if c["cell_type"] != "code":
            continue
        if len(c["source"]) > 1 and "\n" not in texto:
            raise ValueError(f"celula {i}: linhas sem quebra, colapsaria no Jupyter")
        # Linhas de shell (!pip) nao sao Python; viram no-op so para o compile. A
        # indentacao tem de ser preservada, senao um "!pip" dentro de um if vira erro
        # de bloco -- no IPython ele funciona, virando get_ipython().system(...).
        limpo = "\n".join(
            (l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith("!") else l
            for l in texto.split("\n"))
        try:
            compile(limpo, f"<celula {i}>", "exec")
        except SyntaxError as e:
            raise ValueError(f"celula {i}: {e}") from e
    print(f"  validadas {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} "
          "celulas de codigo")


def caderno(celulas: list[dict], nome_kernel: str = "python3") -> dict:
    """Envelope minimo que o Colab aceita."""
    return {
        "cells": celulas,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": nome_kernel, "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
