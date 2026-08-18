"""Testes do backtesting e das metricas.

Duas coisas sao testadas aqui, e por motivos diferentes.

`rolling_origin_splits` decide QUAIS previsoes existem. Um erro de indice nao levanta
excecao: ele muda silenciosamente o numero de janelas, ou deixa uma previsao ver o
proprio alvo. Como todo o resultado do projeto e pareado por janela (bootstrap por
bloco, Diebold-Mariano, comparacao expanding vs deslizante), a garantia de que os dois
modos produzem AS MESMAS origens e pre-requisito do desenho, nao detalhe.

As metricas decidem o que os numeros do manuscrito significam. Sao tres linhas de
codigo cada, e e justamente por isso que ninguem as confere: valor errado aqui sai
publicado com quatro casas decimais e cara de precisao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cv_timeseries.evaluate import mae, rmse, rolling_origin_splits, smape

# desenho real do benchmark
N_MESES, MIN_TREINO, HORIZONTE = 168, 60, 6


@pytest.fixture
def serie() -> pd.Series:
    idx = pd.date_range("2010-01-01", periods=N_MESES, freq="MS")
    return pd.Series(np.arange(float(N_MESES)), index=idx)


# --------------------------------------------------------------------------- #
# rolling_origin_splits
# --------------------------------------------------------------------------- #
def test_reproduz_as_103_janelas_do_desenho_real(serie):
    """103 x 6 = 618 previsoes por modelo e o numero que o manuscrito reporta."""
    janelas = list(rolling_origin_splits(serie, HORIZONTE, MIN_TREINO))
    assert len(janelas) == 103
    assert len(janelas) * HORIZONTE == 618


@pytest.mark.parametrize("n,h,mt,esperado", [
    (168, 6, 60, 103),
    (100, 12, 48, 41),
    (61, 6, 60, 0),      # nao cabe nenhuma: treino 60 + horizonte 6 > 61
    (66, 6, 60, 1),      # cabe exatamente uma
    (67, 6, 60, 2),
])
def test_contagem_de_janelas(n, h, mt, esperado):
    s = pd.Series(np.arange(float(n)), index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    assert len(list(rolling_origin_splits(s, h, mt))) == esperado


def test_teste_sempre_com_o_horizonte_cheio(serie):
    """Janela com teste truncado inflaria a contagem e diluiria a metrica.

    Nota: a guarda `if len(test) == horizon` na implementacao e INALCANCAVEL, porque o
    `range` ja para em `n - horizon`. Confirmado por mutacao: remove-la nao derruba
    nenhum teste. Fica como defesa barata contra mudanca futura no range, e esta
    documentado aqui para ninguem confundir cobertura ausente com codigo morto.
    """
    for _, test in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO):
        assert len(test) == HORIZONTE


def test_treino_e_teste_nao_se_sobrepoem_e_sao_contiguos(serie):
    """Sobreposicao seria o modelo prevendo o que ja viu. Buraco seria gap nao declarado."""
    for train, test in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO):
        assert train.index.max() < test.index.min()
        assert set(train.index).isdisjoint(test.index)
        # contiguidade: o teste comeca no ponto imediatamente seguinte ao treino
        assert test.index[0] == train.index[-1] + pd.DateOffset(months=1)


def test_expanding_cresce_e_comeca_no_inicio(serie):
    tamanhos = []
    for train, _ in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO):
        assert train.index[0] == serie.index[0]
        tamanhos.append(len(train))
    assert tamanhos == sorted(tamanhos)
    assert tamanhos[0] == MIN_TREINO
    assert tamanhos[-1] == N_MESES - HORIZONTE


def test_deslizante_limita_o_treino(serie):
    k = 60
    for train, _ in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=k):
        assert len(train) <= k


def test_deslizante_e_expanding_tem_AS_MESMAS_origens(serie):
    """Pre-requisito do experimento de sensibilidade de janela.

    A comparacao expanding vs deslizante so e pareada se as origens de teste forem
    identicas. Se divergirem, o bootstrap pareado compara janelas diferentes e o
    resultado do experimento perde o sentido, sem que nada acuse.
    """
    exp = [t.index.tolist() for _, t in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO)]
    desl = [t.index.tolist() for _, t in
            rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=60)]
    assert len(exp) == len(desl) == 103
    assert exp == desl


def test_deslizante_maior_que_a_serie_equivale_a_expanding(serie):
    a = [(tr.index.tolist(), te.index.tolist())
         for tr, te in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO)]
    b = [(tr.index.tolist(), te.index.tolist())
         for tr, te in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=10_000)]
    assert a == b


@pytest.mark.parametrize("max_train", [70, 100, 167])
def test_deslizante_maior_que_o_treino_minimo_nao_produz_janela_vazia(serie, max_train):
    """Cobre o clamp em 0 do inicio da fatia, com `max_train_size > min_train_size`.

    Combinacao valida no CLI (--min-train-size 60 --max-train-size 100). Nas primeiras
    janelas `train_end - max_train_size` fica NEGATIVO, e sem o `max(0, ...)` o
    `iloc[-40:60]` do pandas conta a partir do fim da serie e devolve treino VAZIO.
    O modelo entao falha ou preve lixo, so nas janelas iniciais, sem nada acusar.

    Este teste existe porque a suite original passava com o clamp removido: ela so
    exercitava max_train == min_train, onde o indice nunca fica negativo.
    """
    janelas = list(rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=max_train))
    assert len(janelas) == 103
    for train, _ in janelas:
        assert len(train) > 0, "janela com treino vazio"
        assert len(train) >= min(MIN_TREINO, max_train)
        assert len(train) <= max_train


def test_deslizante_com_max_train_maior_comeca_no_inicio_da_serie(serie):
    """Enquanto o historico for menor que o limite, a janela e a serie toda."""
    janelas = list(rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=100))
    primeiro_treino = janelas[0][0]
    assert primeiro_treino.index[0] == serie.index[0]
    assert len(primeiro_treino) == MIN_TREINO


def test_treino_nunca_contem_data_futura(serie):
    for train, test in rolling_origin_splits(serie, HORIZONTE, MIN_TREINO, max_train_size=60):
        assert train.index.max() < test.index.min()
        assert train.max() < test.min()   # valores, nao so datas: a serie e crescente


def test_min_train_maior_que_a_serie_nao_produz_janela(serie):
    assert list(rolling_origin_splits(serie, HORIZONTE, min_train_size=500)) == []


# --------------------------------------------------------------------------- #
# metricas, contra valores calculados a mao
# --------------------------------------------------------------------------- #
def test_mae_valor_conferido_a_mao():
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])
    # |2| + |-2| + |3| = 7, dividido por 3
    assert mae(yt, yp) == pytest.approx(7 / 3)


def test_rmse_valor_conferido_a_mao():
    yt = np.array([10.0, 20.0, 30.0])
    yp = np.array([12.0, 18.0, 33.0])
    # (4 + 4 + 9) / 3 = 17/3, raiz
    assert rmse(yt, yp) == pytest.approx(np.sqrt(17 / 3))


def test_smape_valor_conferido_a_mao():
    yt = np.array([100.0, 100.0])
    yp = np.array([110.0, 90.0])
    # 2*10/210 e 2*10/190, media, vezes 100
    esperado = (2 * 10 / 210 + 2 * 10 / 190) / 2 * 100
    assert smape(yt, yp) == pytest.approx(esperado, rel=1e-6)


def test_previsao_perfeita_zera_as_tres():
    y = np.array([5.0, 10.0, 15.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert smape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_rmse_nunca_menor_que_mae():
    """Desigualdade de normas. Se inverter, uma das duas esta errada."""
    rng = np.random.default_rng(20260817)
    for _ in range(200):
        yt = rng.uniform(1, 10_000, size=50)
        yp = yt + rng.normal(0, 500, size=50)
        assert rmse(yt, yp) >= mae(yt, yp) - 1e-9


def test_rmse_pune_o_outlier_mais_que_o_mae():
    yt = np.full(10, 100.0)
    espalhado = yt + np.full(10, 10.0)          # erro 10 em todos
    concentrado = yt.copy()
    concentrado[0] += 100.0                      # mesmo erro total, num ponto so
    assert mae(yt, espalhado) == pytest.approx(mae(yt, concentrado))
    assert rmse(yt, concentrado) > rmse(yt, espalhado)


def test_metricas_sao_simetricas_na_ordem_dos_argumentos():
    """Trocar y_true por y_pred por engano nao pode mudar o numero em silencio."""
    rng = np.random.default_rng(1)
    yt, yp = rng.uniform(1, 100, 30), rng.uniform(1, 100, 30)
    assert mae(yt, yp) == pytest.approx(mae(yp, yt))
    assert rmse(yt, yp) == pytest.approx(rmse(yp, yt))
    assert smape(yt, yp) == pytest.approx(smape(yp, yt))


def test_smape_fica_entre_0_e_200():
    """Limite superior do sMAPE simetrico. E o que o torna insensivel a divergencia.

    Registrado porque o repo ja foi mordido por isso em outro projeto: o sMAPE
    absorve erro de qualquer magnitude, entao nao denuncia previsao que explode.
    Conferir MAE e RMSE junto nunca e opcional.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        yt = rng.uniform(1, 1000, size=20)
        yp = rng.uniform(1, 1000, size=20)
        assert 0.0 <= smape(yt, yp) <= 200.0 + 1e-9
    # divergencia absurda continua limitada
    assert smape(np.array([1.0]), np.array([1e12])) <= 200.0 + 1e-9


def test_smape_nao_estoura_com_zeros():
    """O eps existe para isto; sem ele seria divisao por zero."""
    z = np.zeros(3)
    assert np.isfinite(smape(z, z))
    assert np.isfinite(smape(z, np.array([0.0, 1.0, 0.0])))


def test_smape_do_pacote_concorda_com_a_forma_usada_nas_analises():
    """Invariante entre modulos, nao dentro de um.

    build_paper_assets.py e run_uncertainty.py recomputam o sMAPE ponto a ponto como
    200*|yp-yt|/(|yt|+|yp|), sem o eps, para poder reamostrar por janela. Se as duas
    formas divergissem, as metricas do manuscrito nao bateriam com as do pipeline, e
    a divergencia apareceria como numero errado publicado, nao como erro.
    """
    rng = np.random.default_rng(42)
    for _ in range(100):
        yt = rng.uniform(5_000, 10_000, size=60)     # escala real da serie
        yp = yt + rng.normal(0, 400, size=60)
        vetorial = float(np.mean(200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))))
        assert smape(yt, yp) == pytest.approx(vetorial, rel=1e-9)
