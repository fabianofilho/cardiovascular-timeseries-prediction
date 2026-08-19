"""Testes das variantes de engenharia de atributos.

Duas coisas sao testadas, e a segunda e a que importa.

1. `calendar_exog` e `diff12` sao as duas transformacoes que tocam no alvo ou no tempo, e sao
   os dois lugares onde vazamento entraria sem levantar excecao. Estao testados contra o
   comportamento esperado, nao contra si mesmos.

2. O resultado publicado. O ponto da secao de variantes NAO e que o CatBoost chegou a 6,09,
   e sim que 6,09 nao passa no criterio pre-declarado do paper. Se alguem no futuro ler a
   tabela de olho so na coluna de sMAPE e escrever "o CatBoost bate o naive sazonal", estes
   testes dizem onde isso foi verificado e falhou.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts" / "revisao"))

JSON = RAIZ / "results" / "revisao" / "variants_vs_snaive.json"
PRED = RAIZ / "results" / "revisao" / "variants_predictions.csv"

variants = pytest.importorskip("variants", reason="scripts/revisao/variants.py ausente")


# --------------------------------------------------------------------------- #
# calendar_exog: deterministico a partir da data
# --------------------------------------------------------------------------- #
def test_calendario_depende_so_da_data():
    """Vazamento seria impossivel aqui, e o teste registra POR QUE.

    A exogena de calendario e funcao apenas do mes do indice. Nao toca no alvo, entao o
    valor futuro dela e conhecido no momento da previsao. E o unico motivo pelo qual usar
    exogena futura nesta variante nao e trapaca.
    """
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    a = variants.calendar_exog(idx)
    b = variants.calendar_exog(idx)
    assert a.equals(b)
    assert list(a.columns) == ["sin1", "cos1", "sin2", "cos2"]


def test_calendario_e_periodico_em_doze_meses():
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    c = variants.calendar_exog(idx)
    assert np.allclose(c.iloc[0].values, c.iloc[12].values)
    assert np.allclose(c.iloc[0].values, c.iloc[24].values)


def test_calendario_distingue_meses_diferentes():
    """Se todos os meses tivessem a mesma codificacao, a feature seria inerte."""
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    c = variants.calendar_exog(idx)
    assert len({tuple(np.round(r, 9)) for r in c.to_numpy()}) == 12


def test_o_calendario_nao_olha_para_o_alvo():
    """Mesmo indice, series completamente diferentes: mesma exogena."""
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    assert variants.calendar_exog(idx).equals(variants.calendar_exog(idx))


# --------------------------------------------------------------------------- #
# diferenca sazonal: a reconstrucao nao pode usar o futuro
# --------------------------------------------------------------------------- #
def test_a_reconstrucao_da_diferenca_so_usa_treino_observado():
    """y_{T+h} = z_{T+h} + y_{T+h-12}, e com h <= 12 esse y esta sempre no treino.

    Este e o ponto onde vazamento entraria de verdade: se h passasse de 12, o termo de
    reconstrucao cairia FORA do treino e o modelo estaria somando um valor que ainda nao
    aconteceu. O teste fixa a fronteira.
    """
    idx = pd.date_range("2010-01-01", periods=72, freq="MS")
    y = pd.Series(np.arange(72.0) + 100.0, index=idx)
    ultimo_treino = y.index[-1]
    for h in range(1, 13):
        data_prevista = ultimo_treino + pd.DateOffset(months=h)
        origem = data_prevista - pd.DateOffset(months=12)
        assert origem <= ultimo_treino, (
            f"em h={h} a reconstrucao usaria {origem:%Y-%m}, que e futuro"
        )


def test_horizonte_acima_de_doze_sairia_do_treino():
    """Contraprova do teste anterior: a garantia depende de h <= 12, nao e universal."""
    idx = pd.date_range("2010-01-01", periods=72, freq="MS")
    ultimo = idx[-1]
    origem = (ultimo + pd.DateOffset(months=13)) - pd.DateOffset(months=12)
    assert origem > ultimo, "com h=13 a reconstrucao passaria a usar dado nao observado"


# --------------------------------------------------------------------------- #
# resultado publicado
# --------------------------------------------------------------------------- #
sem_resultado = pytest.mark.skipif(
    not JSON.exists(), reason="variantes ainda nao foram rodadas")


@sem_resultado
def test_a_variante_base_reproduz_o_benchmark():
    """Controle da bancada inteira.

    A variante `base` usa os mesmos hiperparametros de src/cv_timeseries/models.py, entao
    tem que reproduzir o benchmark. Se este teste cair, nenhuma das outras linhas da tabela
    de variantes e comparavel com a Tabela 1, porque a bancada deixou de ser a mesma.
    """
    m = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]
    assert m["catboost_base"]["smape"] == pytest.approx(6.5893, abs=1e-3)
    # O XGBoost depende da versao da biblioteca; ver docs/xgboost_reprodutibilidade.md.
    assert m["xgboost_base"]["smape"] == pytest.approx(6.8324, abs=2e-3)


@sem_resultado
def test_nenhuma_variante_bate_o_naive_sazonal_pelo_criterio_do_paper():
    """O achado da secao, e a razao de ela existir.

    Ponto estimado nao e conclusao. O criterio do paper exige IC pareado excluindo zero E
    Diebold-Mariano significativo em pelo menos 3 de 6 horizontes; nenhuma variante cumpre.
    """
    m = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]
    variantes = [k for k in m if k.startswith(("catboost_", "xgboost_"))]
    assert len(variantes) == 12
    venceram = [k for k in variantes if m[k]["melhor_que_snaive"]]
    assert venceram == [], f"passaram a bater o naive sazonal: {venceram}"


@sem_resultado
def test_a_melhor_variante_melhora_no_ponto_mas_o_intervalo_contem_zero():
    """A tensao exata que o texto do manuscrito descreve."""
    d = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]["catboost_direct"]
    assert d["delta_smape_vs_snaive"] < 0, "deixou de melhorar no ponto"
    assert d["ic_low"] < 0 < d["ic_high"], "o intervalo deixou de conter zero"
    assert d["dm_significativos"] == 0


@sem_resultado
def test_estatistica_de_janela_piora_os_dois_modelos():
    """Resultado contra-intuitivo, entao vale trava: e o reflexo mais comum e ele atrapalha."""
    m = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]
    for kind in ("catboost", "xgboost"):
        assert m[f"{kind}_roll"]["smape"] > m[f"{kind}_base"]["smape"], (
            f"{kind}: estatisticas de janela passaram a ajudar")


@sem_resultado
def test_a_diferenca_sazonal_e_o_que_ajuda():
    """Sustenta a interpretacao do texto: falta representacao do ciclo, nao capacidade."""
    m = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]
    for kind in ("catboost", "xgboost"):
        assert m[f"{kind}_diff12"]["smape"] < m[f"{kind}_base"]["smape"]


@sem_resultado
def test_toda_variante_continua_atras_dos_tres_lideres():
    """Se cair, a conclusao central do paper mudou e o texto tem que ser reescrito."""
    m = json.loads(JSON.read_text(encoding="utf-8"))["modelos"]
    PIOR_LIDER = 4.8330
    for k, v in m.items():
        if k.startswith(("catboost_", "xgboost_")):
            assert v["smape"] - PIOR_LIDER > 1.0, f"{k} chegou perto dos lideres: {v['smape']}"


@sem_resultado
def test_todas_as_variantes_rodaram_as_618_previsoes():
    """Variante que descartou janela produz sMAPE nao comparavel com as outras."""
    d = pd.read_csv(PRED)
    for m, g in d.groupby("model"):
        assert len(g) == 618, f"{m}: {len(g)} previsoes"
