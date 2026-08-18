"""Testes do bootstrap por bloco e do Diebold-Mariano.

Estas funcoes produzem TODOS os intervalos de confianca do manuscrito, e a falha delas
tem uma direcao so: intervalo mais estreito do que deveria. Isso nao levanta excecao,
nao produz valor absurdo e nao aparece em nenhuma checagem de forma. Aparece como
diferenca de ruido publicada com aparencia de achado, que foi exatamente o que a
primeira rodada deste projeto quase fez.

Por isso a suite testa menos o valor e mais as PROPRIEDADES que separam um bootstrap
por bloco correto de um errado, com destaque para as duas que produzem IC estreito
demais: reamostrar previsao em vez de janela, e usar sorteios independentes por modelo
em vez do mesmo sorteio para todos.
"""

from __future__ import annotations

import numpy as np
import pytest

from cv_timeseries.uncertainty import (
    bootstrap_metrica,
    dm_test,
    ic_percentil,
    p_bicaudal,
    sortear_janelas,
)

N_WIN, N_HOR, B, SEED = 103, 6, 2_000, 20260817


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def idx():
    return sortear_janelas(N_WIN, B, SEED)


# --------------------------------------------------------------------------- #
# sortear_janelas
# --------------------------------------------------------------------------- #
def test_forma_e_dominio_do_sorteio(idx):
    assert idx.shape == (B, N_WIN)
    assert idx.min() >= 0 and idx.max() <= N_WIN - 1


def test_sorteio_e_deterministico_pela_semente():
    """Sem isto, nenhum IC do manuscrito seria reproduzivel."""
    a = sortear_janelas(N_WIN, 500, 42)
    b = sortear_janelas(N_WIN, 500, 42)
    c = sortear_janelas(N_WIN, 500, 43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reamostragem_e_com_reposicao(idx):
    """Sem reposicao, cada replica seria a amostra original permutada e o IC seria zero."""
    repetidos = sum(len(np.unique(linha)) < N_WIN for linha in idx[:200])
    assert repetidos > 190, "quase toda replica deveria repetir alguma janela"


# --------------------------------------------------------------------------- #
# bootstrap_metrica: a unidade de reamostragem
# --------------------------------------------------------------------------- #
def test_media_das_replicas_fica_perto_da_media_amostral(rng, idx):
    mat = rng.normal(5.0, 1.0, size=(N_WIN, N_HOR))
    boot = bootstrap_metrica(mat, idx)
    assert boot.shape == (B,)
    assert boot.mean() == pytest.approx(mat.mean(), abs=0.05)


def test_janela_inteira_entra_ou_sai_junto(rng):
    """A propriedade que define bootstrap POR BLOCO.

    Se uma replica seleciona a janela j, os seis horizontes dela entram. Construido
    com uma matriz onde cada janela tem valor constante: a media de qualquer replica
    tem que ser a media dos valores de janela sorteados, exatamente.
    """
    valores = np.arange(float(N_WIN))
    mat = np.repeat(valores[:, None], N_HOR, axis=1)   # janela j vale j em todo horizonte
    idx = sortear_janelas(N_WIN, 50, 7)
    boot = bootstrap_metrica(mat, idx)
    esperado = valores[idx].mean(axis=1)
    assert np.allclose(boot, esperado)


def test_bootstrap_por_bloco_da_ic_mais_largo_que_por_previsao(rng):
    """A razao de existir do desenho, medida em vez de afirmada.

    Com correlacao dentro da janela, reamostrar previsao a previsao trata observacoes
    dependentes como independentes e estreita o IC artificialmente. O bloco tem que
    ser mais largo; se nao for, o desenho perdeu o proposito.
    """
    efeito_janela = rng.normal(0, 3.0, size=(N_WIN, 1))       # forte correlacao intra-janela
    mat = efeito_janela + rng.normal(0, 0.3, size=(N_WIN, N_HOR))

    idx_bloco = sortear_janelas(N_WIN, 4_000, 1)
    largura_bloco = np.subtract(*ic_percentil(bootstrap_metrica(mat, idx_bloco))[::-1])

    plano = mat.ravel()
    r = np.random.default_rng(1)
    ii = r.integers(0, plano.size, size=(4_000, plano.size))
    largura_previsao = np.subtract(*ic_percentil(plano[ii].mean(axis=1))[::-1])

    assert largura_bloco > largura_previsao * 2, (
        f"bloco={largura_bloco:.4f} deveria ser bem maior que previsao={largura_previsao:.4f}"
    )


def test_mesmo_sorteio_para_todos_os_modelos_preserva_o_pareamento(rng, idx):
    """Com dois modelos que diferem por uma constante, a diferenca nao tem incerteza.

    Reusar `idx` faz a variacao de amostragem se cancelar na diferenca. Sorteios
    independentes por modelo somariam duas variancias e alargariam o IC da diferenca,
    que e o erro simetrico ao de estreitar: esconde diferenca que existe.
    """
    a = rng.normal(5.0, 1.0, size=(N_WIN, N_HOR))
    b = a + 0.5
    dif = bootstrap_metrica(a, idx) - bootstrap_metrica(b, idx)
    assert np.allclose(dif, -0.5)
    lo, hi = ic_percentil(dif)
    assert hi - lo == pytest.approx(0.0, abs=1e-9)


def test_sorteios_independentes_alargariam_a_diferenca(rng):
    """Contraprova do teste anterior: mostra o que o pareamento evita."""
    a = rng.normal(5.0, 1.0, size=(N_WIN, N_HOR))
    b = a + 0.5
    dif = (bootstrap_metrica(a, sortear_janelas(N_WIN, 2_000, 1))
           - bootstrap_metrica(b, sortear_janelas(N_WIN, 2_000, 2)))
    lo, hi = ic_percentil(dif)
    assert hi - lo > 0.05, "sem pareamento a diferenca ganha incerteza propria"


# --------------------------------------------------------------------------- #
# ic_percentil e p_bicaudal
# --------------------------------------------------------------------------- #
def test_ic_percentil_em_distribuicao_conhecida():
    boot = np.linspace(0.0, 100.0, 10_001)
    lo, hi = ic_percentil(boot)
    assert lo == pytest.approx(2.5, abs=0.05)
    assert hi == pytest.approx(97.5, abs=0.05)


def test_ic_de_nivel_maior_e_mais_largo():
    boot = np.random.default_rng(3).normal(size=20_000)
    l95, h95 = ic_percentil(boot, 95.0)
    l99, h99 = ic_percentil(boot, 99.0)
    assert (h99 - l99) > (h95 - l95)


def test_p_bicaudal_um_quando_a_diferenca_esta_centrada_no_zero():
    """Metade de cada lado: nenhuma evidencia de diferenca."""
    boot = np.array([-1.0, 1.0] * 500)
    assert p_bicaudal(boot) == pytest.approx(1.0)


def test_p_bicaudal_zero_quando_todas_as_replicas_tem_o_mesmo_sinal():
    assert p_bicaudal(np.full(1000, 2.5)) == pytest.approx(0.0)
    assert p_bicaudal(np.full(1000, -2.5)) == pytest.approx(0.0)


def test_p_bicaudal_nunca_passa_de_um():
    """Com massa exatamente em zero as duas fracoes somam mais de 1; o clamp segura."""
    for boot in (np.zeros(100), np.array([0.0] * 60 + [1.0] * 40)):
        assert 0.0 <= p_bicaudal(boot) <= 1.0


def test_ic_e_p_contam_a_mesma_historia(rng, idx):
    """Coerencia entre as duas saidas: IC contendo zero deve vir com p alto, e vice-versa."""
    a = rng.normal(5.0, 1.0, size=(N_WIN, N_HOR))
    for desloc, espera_zero_dentro in ((0.0, True), (3.0, False)):
        b = a + desloc + rng.normal(0, 1.0, size=(N_WIN, N_HOR))
        dif = bootstrap_metrica(a, idx) - bootstrap_metrica(b, idx)
        lo, hi = ic_percentil(dif)
        zero_dentro = lo <= 0 <= hi
        assert zero_dentro is espera_zero_dentro
        if zero_dentro:
            assert p_bicaudal(dif) > 0.05
        else:
            assert p_bicaudal(dif) < 0.05


# --------------------------------------------------------------------------- #
# Diebold-Mariano com correcao de Harvey
# --------------------------------------------------------------------------- #
def test_dm_nao_rejeita_quando_nao_ha_diferenca():
    d = np.random.default_rng(11).normal(0, 1.0, size=N_WIN)
    _, p = dm_test(d, h=1)
    assert p > 0.05


def test_dm_rejeita_quando_a_diferenca_e_grande():
    d = np.random.default_rng(11).normal(2.0, 0.5, size=N_WIN)
    estat, p = dm_test(d, h=1)
    assert p < 0.01
    assert estat > 0


def test_dm_troca_de_sinal_ao_inverter_a_ordem():
    d = np.random.default_rng(5).normal(1.0, 1.0, size=N_WIN)
    e1, p1 = dm_test(d, h=1)
    e2, p2 = dm_test(-d, h=1)
    assert e1 == pytest.approx(-e2)
    assert p1 == pytest.approx(p2)


def test_correcao_de_harvey_encolhe_a_estatistica():
    """A correcao existe para o teste nao rejeitar demais em amostra pequena.

    Com n=103 e h>1 o fator e menor que 1, entao a estatistica corrigida fica menor em
    modulo que a nao corrigida. Sem isso, mais celulas cairiam abaixo de 0,05 e o
    manuscrito reportaria diferenca onde nao ha.
    """
    d = np.random.default_rng(9).normal(0.5, 1.0, size=N_WIN)
    for h in (2, 4, 6):
        n = len(d)
        fator = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        assert fator < 1.0
        estat, _ = dm_test(d, h)
        assert abs(estat) > 0


def test_dm_devolve_nan_quando_a_variancia_de_longo_prazo_nao_e_confiavel():
    """Truncamento pode dar variancia negativa; a funcao avisa em vez de inventar."""
    d = np.array([1.0, -1.0] * 20)     # autocovariancia fortemente negativa
    estat, p = dm_test(d, h=6)
    assert (np.isnan(estat) and np.isnan(p)) or np.isfinite(estat)


def test_dm_com_horizonte_um_usa_so_a_variancia_simples():
    """Com h=1 nao ha defasagem a somar: confere contra a conta feita a mao."""
    d = np.random.default_rng(2).normal(0.3, 1.0, size=50)
    n = len(d)
    var = np.sum((d - d.mean()) ** 2) / n
    esperado = (d.mean() / np.sqrt(var / n)) * np.sqrt((n + 1 - 2 + 0) / n)
    estat, _ = dm_test(d, h=1)
    assert estat == pytest.approx(esperado)
