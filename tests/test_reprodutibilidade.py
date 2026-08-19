"""Trava o que a auditoria descobriu sobre reprodutibilidade dos boosters.

Estes testes NAO rodam os modelos: ajustar 103 janelas de XGBoost leva minutos e nao cabe
numa suite. O que eles guardam e a CONSEQUENCIA da descoberta, que e o que se perde primeiro
quando alguem mexe no projeto meses depois:

- a versao do xgboost tem que continuar fixada, porque afrouxa-la muda um numero publicado;
- os resultados medidos em cada ambiente ficam registrados, para que uma execucao futura
  possa ser comparada contra eles em vez de contra a memoria de ninguem.

O achado completo esta em docs/xgboost_reprodutibilidade.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
OPCIONAIS = RAIZ / "requirements-optional.txt"
DOC = RAIZ / "docs" / "xgboost_reprodutibilidade.md"

# sMAPE do XGBoost base nas 103 janelas, por versao da biblioteca. Mesmo codigo, mesmos
# dados, mesma semente: so o ambiente muda. Medido em 2026-08-19.
POR_VERSAO = {
    "2.0.3": 6.854448,
    "2.1.4": 6.854448,
    "3.0.2": 6.832390,
    "3.1.0": 6.832390,
    "3.2.0": 6.832390,
    "3.3.0": 6.984887,
}
GUARDADO_EM_RESULTS = 6.9350


def test_a_versao_do_xgboost_continua_fixada():
    """Regressao de um risco real, nao de um bug.

    `xgboost>=2.0.0` deixava cada maquina resolver para uma versao diferente, e versoes
    diferentes dao sMAPE diferente para o mesmo codigo. Trocar o `==` de volta por `>=`
    reintroduz silenciosamente a irreprodutibilidade da Tabela 1.
    """
    texto = OPCIONAIS.read_text(encoding="utf-8")
    linha = next((l for l in texto.splitlines()
                  if l.strip().startswith("xgboost") and not l.strip().startswith("#")), None)
    assert linha is not None, "xgboost sumiu de requirements-optional.txt"
    assert re.match(r"^xgboost==\d+\.\d+\.\d+$", linha.strip()), (
        f"xgboost precisa estar fixado com '==', esta como {linha.strip()!r}. "
        "O resultado do modelo muda entre versoes; ver docs/xgboost_reprodutibilidade.md."
    )


def test_a_versao_fixada_e_uma_das_medidas():
    """Nao adianta fixar numa versao cujo resultado ninguem mediu."""
    linha = next(l for l in OPCIONAIS.read_text(encoding="utf-8").splitlines()
                 if l.strip().startswith("xgboost=="))
    versao = linha.strip().split("==")[1]
    assert versao in POR_VERSAO, (
        f"xgboost fixado em {versao}, que nao esta na tabela de valores medidos "
        f"({sorted(POR_VERSAO)}). Meça antes de fixar."
    )


def test_o_intervalo_entre_ambientes_nao_muda_conclusao():
    """A amplitude e grande para o digito publicado e pequena para a conclusao.

    Este teste e o que impede alguem de ler a descoberta como "o resultado do XGBoost nao
    vale nada": em TODOS os ambientes medidos ele continua atras do naive sazonal e muito
    atras dos tres lideres, que e o que o paper afirma.
    """
    valores = list(POR_VERSAO.values())
    amplitude = max(valores) - min(valores)
    assert amplitude > 0.1, "se a amplitude sumiu, a tabela foi editada sem medir de novo"

    SEASONAL_NAIVE, MELHOR_LIDER = 6.2693, 4.7015
    for versao, v in POR_VERSAO.items():
        assert v > SEASONAL_NAIVE, f"xgboost {versao} passou o naive sazonal: {v}"
        assert v - MELHOR_LIDER > 2.0, f"xgboost {versao} chegou perto do lider: {v}"


def test_o_valor_guardado_em_results_nao_e_reproduzido_por_nenhuma_versao():
    """Caracterizacao, nao aprovacao. Ver a decisao pendente no doc.

    Enquanto este teste passar, a linha do XGBoost na Tabela 1 vem de um ambiente que nao
    sabemos recriar. Ele passa a falhar no dia em que alguem regenerar o resultado com a
    versao fixada, e essa falha e o lembrete de atualizar o doc e as tabelas derivadas.
    """
    assert all(abs(v - GUARDADO_EM_RESULTS) > 1e-3 for v in POR_VERSAO.values()), (
        "alguma versao passou a reproduzir o valor guardado. Se foi por regeneracao do "
        "resultado, atualize docs/xgboost_reprodutibilidade.md e remova este teste."
    )


@pytest.mark.parametrize("trecho", ["xgboost", "n_jobs", "CatBoost"])
def test_o_achado_esta_documentado(trecho):
    """O numero sozinho nao se explica; o teste aponta para onde ele se explica."""
    assert DOC.exists(), "docs/xgboost_reprodutibilidade.md sumiu"
    assert trecho in DOC.read_text(encoding="utf-8")
