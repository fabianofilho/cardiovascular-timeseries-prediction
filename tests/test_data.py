"""Testes da leitura e agregacao da serie de entrada.

`load_and_aggregate_series` e a primeira peca do pipeline: tudo depois dela herda o que
ela produzir. Um erro aqui nao e sutil como nos outros modulos, ele muda a serie inteira
e apareceria rapido. O que NAO apareceria rapido esta em `test_mes_sem_registro_vira_zero`,
e e o motivo principal deste arquivo existir.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cv_timeseries.data import checa_serie_agregada, load_and_aggregate_series


def escreve(tmp_path: Path, conteudo: str, nome: str = "s.csv") -> str:
    p = tmp_path / nome
    p.write_text(conteudo, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# agregacao
# --------------------------------------------------------------------------- #
def test_soma_as_linhas_do_mesmo_mes(tmp_path):
    """Registro de evento: varias linhas por mes tem que somar, nao substituir."""
    csv = escreve(tmp_path, "date,value\n2020-01-05,10\n2020-01-20,7\n2020-02-03,5\n")
    s = load_and_aggregate_series(csv, "date", "value")
    assert s.loc["2020-01-01"] == 17
    assert s.loc["2020-02-01"] == 5
    assert len(s) == 2


def test_ordena_entrada_desordenada(tmp_path):
    """Nota: o `.sort_index()` da implementacao e REDUNDANTE, porque o `.resample()`
    do pandas ja ordena internamente. Confirmado por mutacao (remove-lo nao derruba
    teste nenhum) e por experimento direto. Fica como defesa barata, e esta
    documentado aqui para ninguem confundir codigo redundante com cobertura ausente.
    O teste continua valendo: garante o CONTRATO de saida ordenada, independente de
    qual linha o cumpre."""
    csv = escreve(tmp_path, "date,value\n2020-03-01,30\n2020-01-01,10\n2020-02-01,20\n")
    s = load_and_aggregate_series(csv, "date", "value")
    assert list(s.index) == list(pd.date_range("2020-01-01", periods=3, freq="MS"))
    assert list(s.values) == [10, 20, 30]


def test_indice_fica_no_primeiro_dia_do_mes(tmp_path):
    """freq MS ancora no inicio do mes; o resto do pipeline assume isso."""
    csv = escreve(tmp_path, "date,value\n2020-01-31,10\n2020-02-29,20\n")
    s = load_and_aggregate_series(csv, "date", "value")
    assert all(d.day == 1 for d in s.index)


def test_a_serie_leva_o_nome_da_coluna_de_valor(tmp_path):
    csv = escreve(tmp_path, "date,obitos\n2020-01-01,10\n")
    s = load_and_aggregate_series(csv, "date", "obitos")
    assert s.name == "obitos"


@pytest.mark.parametrize("freq,esperado", [("MS", 3), ("YS", 1), ("D", 61)])  # 2020 e bissexto
def test_respeita_a_frequencia_pedida(tmp_path, freq, esperado):
    linhas = "\n".join(f"2020-0{m}-{d:02d},1" for m in (1, 2) for d in (1, 15))
    csv = escreve(tmp_path, "date,value\n" + linhas + "\n2020-03-01,1\n")
    s = load_and_aggregate_series(csv, "date", "value", freq=freq)
    assert len(s) == esperado
    assert s.sum() == 5


def test_valor_numerico_em_texto_e_aceito(tmp_path):
    """CSV com aspas ou espaco continua sendo numero."""
    csv = escreve(tmp_path, 'date,value\n2020-01-01," 10 "\n2020-02-01,20\n')
    s = load_and_aggregate_series(csv, "date", "value")
    assert s.loc["2020-01-01"] == 10


# --------------------------------------------------------------------------- #
# o comportamento que precisa ficar visivel
# --------------------------------------------------------------------------- #
def test_mes_sem_registro_vira_zero(tmp_path):
    """CARACTERIZACAO, nao aprovacao: buraco na entrada vira 0, nao NaN.

    O `.sum(min_count=1)` produz NaN para um periodo sem nenhuma linha, e o
    `.fillna(0.0)` seguinte converte esse NaN em zero. Para contagem de eventos isso
    e defensavel quando a ausencia significa mesmo "nao houve evento".

    Para esta serie NAO significa. A mortalidade cardiovascular mensal de Sao Paulo tem
    minimo real de 5.811 obitos; um mes sem linha nenhuma quer dizer falha de extracao,
    nao ausencia de obitos. O zero entra como valor legitimo, vira outlier extremo que
    o modelo tenta ajustar, e nenhuma checagem do pipeline pega: o validador confere
    numero de pontos e soma maior que zero, e as duas passam com um mes zerado no meio.

    Este teste fixa o comportamento atual para que mudanca dele seja deliberada, e
    deixa o risco escrito onde quem mexer vai ler. Ver o item correspondente no TODO.
    """
    csv = escreve(tmp_path, "date,value\n2020-01-15,100\n2020-02-15,110\n2020-04-15,120\n")
    s = load_and_aggregate_series(csv, "date", "value")
    assert len(s) == 4
    assert s.loc["2020-03-01"] == 0.0
    assert not s.isna().any(), "hoje nao sobra NaN: o buraco vira zero"


def test_o_zero_de_buraco_e_indistinguivel_de_zero_real(tmp_path):
    """Consequencia direta do anterior, medida: as duas entradas dao a mesma saida."""
    com_buraco = escreve(tmp_path, "date,value\n2020-01-01,5\n2020-03-01,7\n", "a.csv")
    com_zero = escreve(tmp_path, "date,value\n2020-01-01,5\n2020-02-01,0\n2020-03-01,7\n", "b.csv")
    a = load_and_aggregate_series(com_buraco, "date", "value")
    b = load_and_aggregate_series(com_zero, "date", "value")
    assert list(a.values) == list(b.values) == [5.0, 0.0, 7.0]


# --------------------------------------------------------------------------- #
# validacao de entrada: tem que falhar dizendo o que esta errado
# --------------------------------------------------------------------------- #
def test_coluna_de_data_ausente(tmp_path):
    csv = escreve(tmp_path, "quando,value\n2020-01-01,10\n")
    with pytest.raises(ValueError, match="Coluna de data"):
        load_and_aggregate_series(csv, "date", "value")


def test_coluna_de_valor_ausente(tmp_path):
    csv = escreve(tmp_path, "date,obitos\n2020-01-01,10\n")
    with pytest.raises(ValueError, match="Coluna de valor"):
        load_and_aggregate_series(csv, "date", "value")


def test_data_invalida_falha_em_vez_de_virar_NaT(tmp_path):
    """Sem esta guarda a linha sumiria da agregacao sem aviso."""
    csv = escreve(tmp_path, "date,value\n2020-01-01,10\n30-02-2020,20\n")
    with pytest.raises(ValueError, match="datas inv"):
        load_and_aggregate_series(csv, "date", "value")


def test_valor_nao_numerico_falha(tmp_path):
    csv = escreve(tmp_path, "date,value\n2020-01-01,10\n2020-02-01,indisponivel\n")
    with pytest.raises(ValueError, match="n(a|ã)o num"):
        load_and_aggregate_series(csv, "date", "value")


def test_celula_vazia_conta_como_nao_numerica(tmp_path):
    """Vazio nao pode virar zero em silencio: e o mesmo risco do buraco de mes."""
    csv = escreve(tmp_path, "date,value\n2020-01-01,10\n2020-02-01,\n")
    with pytest.raises(ValueError, match="n(a|ã)o num"):
        load_and_aggregate_series(csv, "date", "value")


def test_arquivo_inexistente_propaga_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_aggregate_series(str(tmp_path / "nao_existe.csv"), "date", "value")


# --------------------------------------------------------------------------- #
# integracao com a serie real do projeto
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[1]
SERIE_REAL = RAIZ / "results" / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"


@pytest.mark.skipif(not SERIE_REAL.exists(), reason="serie versionada ausente")
def test_le_a_serie_real_do_manuscrito():
    """Fecha o ciclo: a funcao produz exatamente a serie que o paper reporta."""
    s = load_and_aggregate_series(str(SERIE_REAL), "date", "value")
    assert len(s) == 168
    assert s.index[0] == pd.Timestamp("2010-01-01")
    assert s.index[-1] == pd.Timestamp("2023-12-01")
    assert s.min() == 5811 and s.max() == 9582
    assert s.mean() == pytest.approx(7246.589, abs=0.01)
    assert (s > 0).all(), "nenhum mes zerado: se aparecer um, ver test_mes_sem_registro_vira_zero"


# --------------------------------------------------------------------------- #
# checa_serie_agregada: a rede que o validador ganhou
# --------------------------------------------------------------------------- #
def _serie(pares):
    idx = pd.to_datetime([d for d, _ in pares])
    return pd.Series([v for _, v in pares], index=idx, dtype=float)


def test_serie_sadia_passa_nas_duas_checagens():
    r = checa_serie_agregada(_serie([
        ("2020-01-01", 100), ("2020-02-01", 110), ("2020-03-01", 120)]))
    assert r["series_no_zero_period"] and r["series_no_date_gap"]
    assert r["_detalhe"]["periodos_zerados"] == []
    assert r["_detalhe"]["periodos_ausentes"] == []


def test_mes_zerado_e_reprovado_e_nomeado():
    """O caso exato que as checagens antigas do validador deixavam passar.

    Numero de pontos e soma maior que zero continuam verdadeiros com um mes zerado no
    meio, e mes zerado em mortalidade e falha de extracao, nao ausencia de obito.
    """
    r = checa_serie_agregada(_serie([
        ("2020-01-01", 100), ("2020-02-01", 0), ("2020-03-01", 120)]))
    assert r["series_no_zero_period"] is False
    assert r["_detalhe"]["periodos_zerados"] == ["2020-02-01"]
    # a serie continua com 3 pontos e soma positiva: por isso as checagens antigas passavam
    assert r["_detalhe"]["n_periodos"] == 3


def test_mes_ausente_do_indice_e_reprovado_e_nomeado():
    r = checa_serie_agregada(_serie([
        ("2020-01-01", 100), ("2020-02-01", 110), ("2020-04-01", 120)]))
    assert r["series_no_date_gap"] is False
    assert r["_detalhe"]["periodos_ausentes"] == ["2020-03-01"]


def test_reporta_todos_os_culpados_nao_so_o_primeiro():
    r = checa_serie_agregada(_serie([
        ("2020-01-01", 0), ("2020-02-01", 110), ("2020-03-01", 0), ("2020-04-01", 0)]))
    assert r["_detalhe"]["periodos_zerados"] == ["2020-01-01", "2020-03-01", "2020-04-01"]


def test_zero_na_borda_tambem_e_pego():
    """Zero no primeiro ou no ultimo periodo e tao suspeito quanto no meio."""
    assert not checa_serie_agregada(_serie([
        ("2020-01-01", 0), ("2020-02-01", 110)]))["series_no_zero_period"]
    assert not checa_serie_agregada(_serie([
        ("2020-01-01", 100), ("2020-02-01", 0)]))["series_no_zero_period"]


def test_detalhe_traz_extremos_para_leitura_rapida():
    r = checa_serie_agregada(_serie([
        ("2020-01-01", 100), ("2020-02-01", 300), ("2020-03-01", 200)]))
    assert r["_detalhe"]["min"] == 100.0 and r["_detalhe"]["max"] == 300.0


@pytest.mark.skipif(not SERIE_REAL.exists(), reason="serie versionada ausente")
def test_a_serie_real_passa_nas_duas_checagens():
    """Se um dia falhar aqui, a extracao quebrou e o benchmark nao deve rodar."""
    s = load_and_aggregate_series(str(SERIE_REAL), "date", "value")
    r = checa_serie_agregada(s)
    assert r["series_no_zero_period"], r["_detalhe"]["periodos_zerados"]
    assert r["series_no_date_gap"], r["_detalhe"]["periodos_ausentes"]
