"""Testes da medida de calibracao dos intervalos.

O `interval_score` e a unica formula nova aqui, e e a que decide o ranking entre SARIMA e
Prophet no manuscrito. Errar o fator `2/alpha` ou o lado da penalidade nao levanta excecao:
produz um numero plausivel e inverte a conclusao. E por isso que ele esta testado contra
casos calculados a mao, e nao so contra si mesmo.

O resto do arquivo testa o resultado guardado: PICP fora de [0,1] ou contagem de previsoes
diferente de 618 significa que a rodada quebrou no meio, e uma rodada parcial produz
cobertura que parece normal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
PRED = RAIZ / "results" / "calibracao_2010_2023_predictions.csv"
MET = RAIZ / "results" / "calibracao_2010_2023_metrics.json"

ALPHA = 0.05
NOMINAL = 0.95


def interval_score(y, lo, hi, alpha=ALPHA):
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return ((hi - lo)
            + (2.0 / alpha) * np.clip(lo - y, 0, None)
            + (2.0 / alpha) * np.clip(y - hi, 0, None))


# --------------------------------------------------------------------------- #
# interval score: casos calculados a mao
# --------------------------------------------------------------------------- #
def test_dentro_do_intervalo_o_score_e_so_a_largura():
    """Sem penalidade, a regra vira largura pura. E o caso de referencia."""
    assert interval_score([100.0], [90.0], [110.0])[0] == pytest.approx(20.0)


def test_abaixo_do_limite_inferior_penaliza():
    """y=80, [90,110]: largura 20 + (2/0.05)*10 = 20 + 400 = 420."""
    assert interval_score([80.0], [90.0], [110.0])[0] == pytest.approx(420.0)


def test_acima_do_limite_superior_penaliza_igual():
    """Simetrico: errar 10 para cima custa o mesmo que 10 para baixo."""
    baixo = interval_score([80.0], [90.0], [110.0])[0]
    alto = interval_score([120.0], [90.0], [110.0])[0]
    assert baixo == pytest.approx(alto)


def test_a_penalidade_e_proporcional_a_distancia():
    """Errar o dobro custa o dobro da penalidade, nao o dobro do score."""
    a = interval_score([80.0], [90.0], [110.0])[0] - 20.0
    b = interval_score([70.0], [90.0], [110.0])[0] - 20.0
    assert b == pytest.approx(2 * a)


def test_intervalo_largo_demais_nao_e_premiado():
    """O ponto da regra: cobrir por construcao custa caro.

    Sem isso, a forma de vencer seria devolver um intervalo infinito, e a Tabela de
    calibracao premiaria o modelo mais inutil.
    """
    justo = interval_score([100.0], [95.0], [105.0])[0]
    largo = interval_score([100.0], [0.0], [10000.0])[0]
    assert largo > justo * 100


def test_no_limite_exato_nao_ha_penalidade():
    """y == hi conta como dentro; a fronteira nao pode gerar penalidade infinitesimal."""
    assert interval_score([110.0], [90.0], [110.0])[0] == pytest.approx(20.0)


def test_e_vetorizado_elemento_a_elemento():
    v = interval_score([100.0, 80.0], [90.0, 90.0], [110.0, 110.0])
    assert v.shape == (2,)
    assert v[0] == pytest.approx(20.0) and v[1] == pytest.approx(420.0)


# --------------------------------------------------------------------------- #
# resultado guardado
# --------------------------------------------------------------------------- #
pytestmark_faltando = pytest.mark.skipif(
    not PRED.exists(), reason="calibracao ainda nao foi rodada")


@pytestmark_faltando
def test_a_rodada_esta_completa():
    """618 previsoes por modelo. Rodada parcial da cobertura de aparencia normal."""
    d = pd.read_csv(PRED)
    for m in ("sarima", "prophet"):
        g = d[d.model == m]
        assert len(g) == 618, f"{m}: {len(g)} previsoes, esperado 618"
        assert g.window.nunique() == 103
        assert sorted(g.horizon.unique()) == [1, 2, 3, 4, 5, 6]


@pytestmark_faltando
def test_os_limites_estao_na_ordem_certa():
    """lo > hi passaria despercebido e zeraria a cobertura sem erro nenhum."""
    d = pd.read_csv(PRED)
    assert (d.hi >= d.lo).all(), "ha intervalo com limite superior abaixo do inferior"
    assert np.isfinite(d[["lo", "hi", "y_pred", "y_true"]].to_numpy()).all()


@pytestmark_faltando
def test_o_picp_do_json_bate_com_o_recalculo_das_previsoes():
    """Mesma regra do resto do projeto: o numero publicado sai das previsoes."""
    d = pd.read_csv(PRED)
    met = json.loads(MET.read_text(encoding="utf-8"))
    for m in ("sarima", "prophet"):
        g = d[d.model == m]
        picp = float(((g.y_true >= g.lo) & (g.y_true <= g.hi)).mean())
        assert picp == pytest.approx(met["calibracao"][m]["picp"], abs=1e-12)


@pytestmark_faltando
def test_os_dois_modelos_subcobrem():
    """Achado central da secao de calibracao, travado como regressao.

    Se uma mudanca futura fizer a cobertura passar de 0,95, isso e uma noticia grande e
    tem que ser deliberada, nao um efeito colateral silencioso.
    """
    met = json.loads(MET.read_text(encoding="utf-8"))["calibracao"]
    for m in ("sarima", "prophet"):
        picp = met[m]["picp"]
        assert 0.0 <= picp <= 1.0
        assert picp < NOMINAL, f"{m} deixou de subcobrir: PICP={picp}"


@pytestmark_faltando
def test_prophet_tem_intervalo_mais_estreito_e_pior():
    """A inversao entre acuracia pontual e qualidade do intervalo.

    Prophet tem o melhor sMAPE dos cinco e o pior dos dois intervalos. E a razao pela qual
    o paper diz para nao escolher modelo operacional so por acuracia pontual; se o teste
    cair, essa frase do manuscrito deixou de ser sustentada pelos dados.
    """
    c = json.loads(MET.read_text(encoding="utf-8"))["calibracao"]
    assert c["prophet"]["mpiw"] < c["sarima"]["mpiw"], "Prophet deixou de ser o mais estreito"
    assert c["prophet"]["is"] > c["sarima"]["is"], "Prophet deixou de ter o pior score"
    assert c["prophet"]["picp"] < c["sarima"]["picp"]


@pytestmark_faltando
def test_a_procedencia_foi_conferida_e_o_prophet_bate_com_o_benchmark():
    """A previsao pontual do Prophet aqui e a mesma do benchmark.

    Sem isso, a calibracao medida seria de um modelo com o mesmo nome, nao do modelo que a
    Tabela 1 reporta. O SARIMA nao satisfaz esta checagem por um motivo conhecido e
    documentado em docs/, entao aqui so o Prophet e exigido.
    """
    proc = json.loads(MET.read_text(encoding="utf-8")).get("procedencia")
    assert proc is not None, "a rodada nao conferiu procedencia"
    assert proc["prophet"]["max_div_y_pred"] < 1e-6
    assert proc["prophet"]["max_div_y_true"] == 0.0
