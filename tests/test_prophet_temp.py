"""Testes da rodada de temperatura no Prophet e do resultado do TabPFN.

Os dois entram no manuscrito e os dois vieram de fora, entao o que esta travado aqui e o
que sustenta cada afirmacao, e nao o numero em si:

- a rodada do Prophet SEM temperatura precisa reproduzir a linha do Prophet na Tabela 1,
  senao o efeito medido seria a diferenca entre duas implementacoes de Prophet e nao o
  efeito da covariavel;
- a bancada da rodada do TabPFN precisa reproduzir as referencias ja publicadas, pelo mesmo
  motivo;
- e a afirmacao de que o TabPFN bate as referencias ingenuas NAO pode passar a ser feita
  enquanto o teste pareado nao existir. Ha um teste para isso.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PT = RAIZ / "results" / "revisao" / "prophet_temp_vs_base.json"
# A rodada de 23/09 nas 103 janelas. A v4 continua no repositorio como historico, e
# apontar para ela aqui era o que deixava estes testes verdes sobre dado velho.
TAB = RAIZ / "results" / "revisao" / "tabpfn_resultados_v5_2026-09-23.json"
PRED_TAB = RAIZ / "results" / "revisao" / "tabpfn_predictions.csv"
VER = RAIZ / "paper" / "verified_numbers.json"

sem_pt = pytest.mark.skipif(not PT.exists(), reason="rodada do Prophet com temperatura ausente")
sem_tab = pytest.mark.skipif(not TAB.exists(), reason="resultado do TabPFN ausente")


# --------------------------------------------------------------------------- #
# Prophet com temperatura
# --------------------------------------------------------------------------- #
@sem_pt
def test_a_rodada_sem_temperatura_reproduz_a_tabela1():
    """Procedencia. Sem isto a Tabela 5 compararia dois Prophets diferentes.

    O manuscrito afirma explicitamente que esta checagem foi feita; se ela deixar de
    valer, a afirmacao no texto passa a ser falsa.
    """
    meta = json.loads(PT.read_text(encoding="utf-8"))["_meta"]
    assert meta["procedencia_max_div"] < 1e-6, (
        f"a rodada sem temperatura divergiu da Tabela 1 em {meta['procedencia_max_div']}")


@sem_pt
def test_prophet_de_fato_aceita_a_covariavel():
    """O manuscrito dizia que nao aceita. A rodada existir ja e a refutacao."""
    m = json.loads(PT.read_text(encoding="utf-8"))["modelos"]
    assert "prophet_temp" in m and m["prophet_temp"]["smape"] > 0


@sem_pt
def test_a_temperatura_piora_o_prophet_mas_nao_de_forma_significativa():
    """As duas metades importam.

    So a primeira viraria "a temperatura prejudica o Prophet", que os dados nao sustentam.
    So a segunda perderia o unico caso dos quatro em que o sinal se inverte.
    """
    d = json.loads(PT.read_text(encoding="utf-8"))["modelos"]["prophet_temp"]
    assert d["ganho_pp"] < 0, "a temperatura deixou de piorar o Prophet"
    assert d["ic_low"] < 0 < d["ic_high"], "o intervalo deixou de conter zero"
    assert d["dm_significativos"] == 0


@sem_pt
def test_o_teto_com_vazamento_nao_atinge_o_criterio():
    """Ate com a temperatura futura observada o ganho nao fecha o criterio do paper."""
    d = json.loads(PT.read_text(encoding="utf-8"))["modelos"]["prophet_temp_ceiling"]
    assert d["dm_significativos"] < 3, "o teto passou a atender o criterio; reescreva o texto"


@sem_pt
def test_prophet_continua_o_melhor_mesmo_com_a_covariavel():
    """Se cair, a Tabela 1 e a Tabela 5 passam a contar historias diferentes."""
    m = json.loads(PT.read_text(encoding="utf-8"))["modelos"]
    v = json.loads(VER.read_text(encoding="utf-8"))["tabela1"]
    assert m["prophet_temp"]["smape"] < v["sarima"]["smape"]
    assert m["prophet_temp"]["smape"] < v["catboost"]["smape"]


# --------------------------------------------------------------------------- #
# TabPFN
# --------------------------------------------------------------------------- #
def _tabpfn():
    r = json.loads(TAB.read_text(encoding="utf-8"))["resultados"]
    return {x["modelo"]: x for x in r}


@sem_tab
def test_a_bancada_do_tabpfn_reproduz_o_que_ja_estava_publicado():
    """Controle. Quatro modelos batem exato; e o que autoriza ler a linha nova."""
    r = _tabpfn()
    for m in ("naive", "snaive", "snaive_drift", "catboost"):
        assert r[m]["diferenca_pp"] == pytest.approx(0.0, abs=1e-9), (
            f"{m} deixou de bater: {r[m]}")


@sem_tab
def test_o_tabpfn_e_o_melhor_tabular_testado():
    """Afirmacao que o manuscrito faz, e que a estimativa pontual sustenta."""
    r = _tabpfn()
    assert r["tabpfn"]["smape_obtido"] < r["catboost"]["smape_obtido"]
    assert r["tabpfn"]["smape_obtido"] < r["xgboost"]["smape_obtido"]


@sem_tab
def test_o_tabpfn_continua_atras_dos_tres_lideres():
    r = _tabpfn()
    v = json.loads(VER.read_text(encoding="utf-8"))["tabela1"]
    for lider in ("prophet", "sarima", "timesfm"):
        assert r["tabpfn"]["smape_obtido"] - v[lider]["smape"] > 1.0


@sem_tab
def test_a_vitoria_sobre_o_naive_foi_medida_e_nao_atende_o_criterio():
    """O inverso do teste que estava aqui, pela mesma razao que ele existia.

    Enquanto nao havia previsao por janela, o teste guardava a comparacao que mostrava
    que a afirmacao de vitoria nao tinha suporte. A medicao foi feita, e o resultado nao
    e o previsto: o intervalo pareado EXCLUI zero, o que nenhuma variante de boosting
    conseguiu, e mesmo assim o criterio nao e atendido, porque o Diebold-Mariano fica em
    1 de 6. Agora o teste cobra o resultado medido, e reprova se alguem escrever vitoria.
    """
    vj = RAIZ / "results" / "revisao" / "variants_vs_snaive.json"
    if not vj.exists():
        pytest.skip("variantes ainda nao rodadas")
    d = json.loads(vj.read_text(encoding="utf-8"))["modelos"]
    assert "tabpfn" in d, (
        "o TabPFN sumiu do variants_vs_snaive.json. O criterio pre-declarado so pode ser "
        "avaliado com a previsao janela a janela; sem ela a afirmacao volta a nao ter "
        "suporte.")
    tab = d["tabpfn"]

    assert tab["ic_high"] < 0, (
        f"o intervalo pareado contra o naive sazonal deixou de excluir zero: "
        f"[{tab['ic_low']:.3f}, {tab['ic_high']:.3f}]. O texto afirma que exclui.")
    assert tab["dm_significativos"] < 3, (
        f"o Diebold-Mariano passou a {tab['dm_significativos']}/6 e o criterio agora e "
        "atendido. Isso muda a conclusao do manuscrito: reescreva a subsecao do modelo "
        "tabular antes de deixar o teste verde.")
    assert tab["melhor_que_snaive"] is False


@sem_tab
def test_a_previsao_por_janela_existe_e_esta_completa():
    """Sem as 103x6 linhas o criterio nao e calculavel, e foi assim por tres semanas."""
    if not PRED_TAB.exists():
        pytest.skip("CSV por janela ausente")
    linhas = PRED_TAB.read_text(encoding="utf-8").strip().split("\n")
    assert len(linhas) - 1 == 103 * 6, (
        f"{len(linhas) - 1} previsoes, esperado {103 * 6}. O teste pareado sobre um "
        "conjunto incompleto nao e comparavel com o dos outros modelos.")


@sem_tab
def test_o_resultado_esta_documentado():
    doc = RAIZ / "docs" / "tabpfn.md"
    assert doc.exists(), "docs/tabpfn.md sumiu"
    texto = doc.read_text(encoding="utf-8")
    for trecho in ("critério pré-declarado", "não atende", "tabpfn_client"):
        assert trecho in texto, (
            f"{trecho!r} saiu de docs/tabpfn.md. O veredito medido e a ressalva sobre a "
            "versao do cliente sao o que impede alguem de reportar o numero sozinho.")
