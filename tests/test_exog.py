"""Testes da politica de exogena, com foco em vazamento.

Por que estes testes existem, e nao outros: vazamento de exogena e a falha que este
projeto tem mais a perder e menos chance de perceber. Ele nao levanta excecao, nao
deixa NaN e nao quebra nenhuma metrica de forma. Ele so melhora o resultado, e
resultado que melhora sozinho ninguem investiga. O sintoma aparece na revisao por
pares, ou nunca.

A tecnica dos testes e sempre a mesma: usar uma exogena ESTRITAMENTE CRESCENTE, na
qual todo valor posterior ao treino e maior que qualquer valor visivel. Assim
"vazou" vira uma comparacao numerica simples, em vez de depender de inspecionar
indice ou confiar na leitura do codigo.

O desenho exercitado e o do benchmark real: 168 meses, treino minimo de 60, horizonte
de 6, o que da 103 janelas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cv_timeseries.exog import POLITICAS, build_exog_frames

N_MESES = 168
MIN_TREINO = 60
HORIZONTE = 6
N_JANELAS = N_MESES - HORIZONTE - MIN_TREINO + 1  # 103


@pytest.fixture
def exog_crescente() -> pd.DataFrame:
    """Exogena onde o valor e o proprio indice: qualquer futuro e maior que o passado."""
    idx = pd.date_range("2010-01-01", periods=N_MESES, freq="MS")
    return pd.DataFrame({"tmin": np.arange(float(N_MESES))}, index=idx)


def janelas(exog: pd.DataFrame):
    """As mesmas origens do rolling origin do benchmark."""
    for fim in range(MIN_TREINO, N_MESES - HORIZONTE + 1):
        yield exog.index[:fim], exog.index[fim:fim + HORIZONTE]


def test_desenho_tem_as_103_janelas(exog_crescente):
    assert sum(1 for _ in janelas(exog_crescente)) == N_JANELAS == 103


@pytest.mark.parametrize("policy", ["climatology", "lag12"])
def test_sem_vazamento_em_todas_as_janelas(exog_crescente, policy):
    """O teste central: nenhuma politica operacional pode ver valor pos-treino.

    Roda nas 103 janelas, nao numa amostra: vazamento que aparece so em janela
    especifica (a primeira, a ultima, uma virada de ano) e exatamente o que passa
    despercebido num teste de caso unico.
    """
    for train_idx, test_idx in janelas(exog_crescente):
        _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, policy)
        maior_visivel = exog_crescente.loc[train_idx, "tmin"].max()
        assert fut["tmin"].max() <= maior_visivel + 1e-9, (
            f"politica {policy} usou valor posterior ao treino na janela que termina "
            f"em {train_idx[-1]:%Y-%m}"
        )


def test_observed_vaza_de_proposito(exog_crescente):
    """O cenario-teto TEM que vazar; se parar de vazar, deixou de medir o teto."""
    vazou = 0
    for train_idx, test_idx in janelas(exog_crescente):
        _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, "observed")
        esperado = exog_crescente.loc[test_idx, "tmin"].to_numpy()
        assert np.allclose(fut["tmin"].to_numpy(), esperado)
        if fut["tmin"].max() > exog_crescente.loc[train_idx, "tmin"].max():
            vazou += 1
    assert vazou == N_JANELAS


def test_exog_train_nunca_passa_do_fim_do_treino(exog_crescente):
    for train_idx, test_idx in janelas(exog_crescente):
        tr, _ = build_exog_frames(exog_crescente, train_idx, test_idx, "climatology")
        assert tr.index.max() == train_idx.max()
        assert len(tr) == len(train_idx)


def test_climatologia_e_recalculada_por_janela(exog_crescente):
    """Se fosse calculada uma vez sobre a serie toda, seria vazamento silencioso.

    Numa exogena crescente, a media do mes-do-ano cresce conforme o treino avanca.
    Valores identicos entre janelas distantes denunciariam calculo global.
    """
    vistos = []
    for fim in (60, 100, 140):
        train_idx = exog_crescente.index[:fim]
        test_idx = exog_crescente.index[fim:fim + HORIZONTE]
        _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, "climatology")
        vistos.append(fut["tmin"].to_numpy())
    assert not np.allclose(vistos[0], vistos[1])
    assert not np.allclose(vistos[1], vistos[2])
    # e cresce, porque o treino so ganha meses com valor maior
    assert vistos[0].mean() < vistos[1].mean() < vistos[2].mean()


def test_climatologia_usa_a_media_do_mes_do_ano(exog_crescente):
    """Confere o valor, nao so a ausencia de vazamento: politica certa, conta certa."""
    train_idx = exog_crescente.index[:120]        # 10 anos exatos
    test_idx = exog_crescente.index[120:126]
    _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, "climatology")
    truncada = exog_crescente.loc[train_idx]
    for d in test_idx:
        esperado = truncada[truncada.index.month == d.month]["tmin"].mean()
        assert fut.loc[d, "tmin"] == pytest.approx(esperado)


def test_lag12_usa_exatamente_o_valor_de_doze_meses_antes(exog_crescente):
    train_idx = exog_crescente.index[:120]
    test_idx = exog_crescente.index[120:126]
    _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, "lag12")
    for d in test_idx:
        esperado = exog_crescente.loc[d - pd.DateOffset(months=12), "tmin"]
        assert fut.loc[d, "tmin"] == pytest.approx(esperado)


@pytest.mark.parametrize("horizonte", [13, 18, 24])
def test_lag12_falha_explicito_quando_o_horizonte_passa_de_doze(exog_crescente, horizonte):
    """Com horizonte > 12 a defasagem alcancaria o futuro. Tem que falhar dizendo isso.

    Antes da guarda, o pandas levantava KeyError apontando para a data, sem dizer que
    a causa era vazamento. Erro que nao nomeia a causa vira "contorna com try/except".
    """
    train_idx = exog_crescente.index[:100]
    test_idx = exog_crescente.index[100:100 + horizonte]
    with pytest.raises(ValueError, match="lag12"):
        build_exog_frames(exog_crescente, train_idx, test_idx, "lag12")


@pytest.mark.parametrize("horizonte", [1, 6, 12])
def test_lag12_aceita_horizonte_ate_doze(exog_crescente, horizonte):
    train_idx = exog_crescente.index[:100]
    test_idx = exog_crescente.index[100:100 + horizonte]
    _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, "lag12")
    assert len(fut) == horizonte


def test_politica_desconhecida_falha(exog_crescente):
    train_idx = exog_crescente.index[:60]
    test_idx = exog_crescente.index[60:66]
    with pytest.raises(ValueError, match="desconhecida"):
        build_exog_frames(exog_crescente, train_idx, test_idx, "climatologia")


def test_todas_as_politicas_declaradas_funcionam(exog_crescente):
    """POLITICAS e o que o CLI oferece; nenhuma pode estar quebrada."""
    train_idx = exog_crescente.index[:60]
    test_idx = exog_crescente.index[60:66]
    for policy in POLITICAS:
        tr, fut = build_exog_frames(exog_crescente, train_idx, test_idx, policy)
        assert list(fut.index) == list(test_idx)
        assert not fut.isna().any().any(), f"{policy} produziu NaN"
        assert list(fut.columns) == list(exog_crescente.columns)


def test_janela_deslizante_nao_afeta_a_regra(exog_crescente):
    """Com --max-train-size o treino encurta, mas o corte anti-vazamento e o mesmo.

    A truncagem usa o FIM do treino, nao o comeco, entao janela deslizante nao pode
    abrir brecha. Testa porque o desenho deslizante entrou depois da regra.
    """
    for fim in range(MIN_TREINO, N_MESES - HORIZONTE + 1, 7):
        train_idx = exog_crescente.index[max(0, fim - 60):fim]   # deslizante de 60
        test_idx = exog_crescente.index[fim:fim + HORIZONTE]
        for policy in ("climatology", "lag12"):
            _, fut = build_exog_frames(exog_crescente, train_idx, test_idx, policy)
            assert fut["tmin"].max() <= exog_crescente.loc[train_idx, "tmin"].max() + 1e-9


def test_multiplas_colunas_de_exogena(exog_crescente):
    """O CLI aceita --exog-cols com varias colunas; nenhuma pode ser perdida."""
    ex = exog_crescente.copy()
    ex["tmed"] = ex["tmin"] * 2 + 1
    train_idx = ex.index[:120]
    test_idx = ex.index[120:126]
    for policy in POLITICAS:
        tr, fut = build_exog_frames(ex, train_idx, test_idx, policy)
        assert list(fut.columns) == ["tmin", "tmed"]
        assert list(tr.columns) == ["tmin", "tmed"]
