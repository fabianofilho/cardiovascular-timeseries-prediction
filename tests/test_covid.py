"""Testes da sensibilidade ao periodo COVID.

O recorte serve para responder uma pergunta unica: a conclusao central do paper depende da
pandemia? Os testes aqui guardam as tres coisas que fazem a resposta valer alguma coisa:

- o recorte "completo" e CONTROLE e tem que reproduzir a Tabela 1. Sem isso, os outros dois
  recortes medem outra coisa e nao ha como saber o que;
- o desenho exclui JANELAS INTEIRAS, nao datas soltas. Excluir datas quebraria o retangulo
  janela x horizonte que o bootstrap pareado exige, e o teste passaria a comparar amostras
  de tamanhos diferentes sem avisar;
- o veredito, em cada recorte, sai do criterio COMPLETO. So o intervalo de bootstrap daria
  uma resposta diferente e mais generosa nos boosters, e e por isso que ele nao basta.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
JSON = RAIZ / "results" / "revisao" / "covid_sensibilidade.json"
VER = RAIZ / "paper" / "verified_numbers.json"

sem_dados = pytest.mark.skipif(not JSON.exists(), reason="sensibilidade ao COVID nao rodada")


def _rec():
    return json.loads(JSON.read_text(encoding="utf-8"))["recortes"]


@sem_dados
def test_o_recorte_completo_reproduz_a_tabela1():
    """Controle. Se cair, nenhum outro numero deste arquivo significa alguma coisa."""
    meta = json.loads(JSON.read_text(encoding="utf-8"))["_meta"]
    assert meta["controle_max_div"] < 1e-6, (
        f"o recorte completo divergiu da Tabela 1 em {meta['controle_max_div']}")


@sem_dados
def test_o_completo_usa_as_103_janelas():
    assert _rec()["completo"]["n_janelas"] == 103


@sem_dados
def test_os_recortes_removem_janelas_inteiras():
    """O desenho precisa manter o retangulo janela x horizonte.

    Se alguem trocar a exclusao de janelas por exclusao de datas, a contagem de previsoes
    deixa de ser multiplo de 6 e o reshape do bootstrap quebra ou, pior, silenciosamente
    compara amostras diferentes.
    """
    r = _rec()
    for nome in ("sem_covid_agudo", "sem_covid_amplo"):
        nw = r[nome]["n_janelas"]
        assert 0 < nw < 103, f"{nome}: {nw} janelas"
    assert r["sem_covid_amplo"]["n_janelas"] < r["sem_covid_agudo"]["n_janelas"], (
        "o recorte amplo tem que remover mais janelas que o agudo")


@sem_dados
def test_nenhum_par_dos_lideres_se_separa_em_nenhum_recorte():
    """O achado central do paper, testado contra a pandemia.

    Este e o teste que responde "e se a COVID estiver dominando a comparacao?". Se um par
    passar a se separar em algum recorte, o paper precisa dizer isso, e a frase atual sobre
    robustez fica falsa.
    """
    for nome, r in _rec().items():
        for par, d in r["pares_top3"].items():
            assert d["distinguiveis"] is False, (
                f"{nome}: {par} passou a se separar ({d})")
            assert d["dm_significativos"] == 0, f"{nome}: {par} DM {d['dm_significativos']}/6"


@sem_dados
def test_excluir_a_pandemia_melhora_todo_mundo():
    """Assinatura de periodo dificil para todos, e nao de periodo que favorece um metodo.

    E o que autoriza a leitura do texto. Se a melhora fosse concentrada num modelo, a
    conclusao sobre a pandemia seria outra.
    """
    r = _rec()
    for m in ("prophet", "sarima", "timesfm", "catboost", "xgboost", "snaive"):
        assert r["sem_covid_agudo"]["smape"][m] < r["completo"]["smape"][m], m
        assert r["sem_covid_amplo"]["smape"][m] < r["completo"]["smape"][m], m


@sem_dados
def test_a_desvantagem_do_boosting_aumenta_sem_a_pandemia():
    """Registrado porque e contra-intuitivo: a pandemia ESCONDIA parte do problema."""
    r = _rec()
    for m in ("catboost", "xgboost"):
        completo = r["completo"]["boosting_vs_snaive"][m]["delta"]
        agudo = r["sem_covid_agudo"]["boosting_vs_snaive"][m]["delta"]
        assert agudo > completo > 0, f"{m}: completo {completo}, agudo {agudo}"


@sem_dados
def test_mesmo_assim_a_desvantagem_nao_atinge_o_criterio():
    """A metade que o bootstrap sozinho nao veria.

    Sem o Diebold-Mariano, o intervalo excluindo zero levaria a escrever "o XGBoost e
    significativamente pior que o naive sazonal", que o criterio do paper nao sustenta.
    Este teste guarda essa distincao.
    """
    for nome, r in _rec().items():
        for m, d in r["boosting_vs_snaive"].items():
            assert d["pior_que_snaive"] is False, (
                f"{nome}/{m} passou a atender o criterio: {d}. "
                "Se e real, reescreva a secao; nao deixe o texto desatualizado.")
            assert d["dm_significativos"] < 3


@sem_dados
def test_o_intervalo_sozinho_daria_outra_resposta():
    """Caracterizacao da armadilha, para ela nao voltar.

    No recorte agudo o intervalo do XGBoost exclui zero, e so o DM impede a conclusao.
    """
    d = _rec()["sem_covid_agudo"]["boosting_vs_snaive"]["xgboost"]
    assert d["ic_low"] > 0, "o intervalo deixou de excluir zero"
    assert d["dm_significativos"] < 3, "o DM passou a acompanhar; reescreva o texto"
