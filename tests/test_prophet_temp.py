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
TAB = RAIZ / "results" / "revisao" / "tabpfn_resultados_v4.json"
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
def test_a_vitoria_sobre_o_naive_nao_pode_ser_afirmada_sem_o_teste_pareado():
    """O ponto do teste, e a razao de ele existir.

    A margem do TabPFN sobre o naive sazonal com drift e MENOR que a do catboost_direct,
    que reprovou no criterio pre-declarado. Enquanto nao houver previsoes por janela, a
    afirmacao de vitoria nao tem suporte, e este teste guarda a comparacao que mostra isso.
    """
    r = _tabpfn()
    margem_tabpfn = r["snaive_drift"]["smape_obtido"] - r["tabpfn"]["smape_obtido"]

    vj = RAIZ / "results" / "revisao" / "variants_vs_snaive.json"
    if not vj.exists():
        pytest.skip("variantes ainda nao rodadas")
    d = json.loads(vj.read_text(encoding="utf-8"))["modelos"]["catboost_direct"]
    margem_reprovada = -d["delta_smape_vs_snaive"]

    assert margem_tabpfn < margem_reprovada, (
        f"a margem do TabPFN ({margem_tabpfn:.3f} pp) passou a ser maior que a do "
        f"catboost_direct ({margem_reprovada:.3f} pp), que reprovou no criterio. "
        "Se isso mudou, rode o teste pareado antes de afirmar vitoria."
    )
    assert d["melhor_que_snaive"] is False


@sem_tab
def test_o_pendente_esta_documentado():
    doc = RAIZ / "docs" / "tabpfn.md"
    assert doc.exists(), "docs/tabpfn.md sumiu"
    texto = doc.read_text(encoding="utf-8")
    for trecho in ("previsões por janela", "critério pré-declarado", "Isabella Saade"):
        assert trecho in texto
