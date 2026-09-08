"""Testes das duas checagens de robustez que fecham a rodada de revisao.

Ambas testam AFIRMACOES DO PAPER, e nao numeros novos:

- `check_dm_variance` testa se o truncamento em h-1 usado no teste de Diebold-Mariano
  subestima a variancia. Se subestimasse, o teste rejeitaria demais, e as celulas "0 de 6"
  espalhadas pelo manuscrito seriam otimismo do estimador e nao ausencia de diferenca.
- `run_temp_melhorado` testa o MECANISMO que o texto atribui ao ganho da temperatura nos
  boosters. E uma predicao falsificavel: se dar calendario explicito nao encolher o ganho,
  a frase do manuscrito esta errada e precisa sair.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
DM = RAIZ / "results" / "revisao" / "dm_variancia.json"
TEMP = RAIZ / "results" / "revisao" / "temp_melhorado.csv"

sem_dm = pytest.mark.skipif(not DM.exists(), reason="checagem de variancia do DM nao rodada")
sem_temp = pytest.mark.skipif(not TEMP.exists(), reason="teste do mecanismo nao rodado")


# --------------------------------------------------------------------------- #
# variancia do Diebold-Mariano
# --------------------------------------------------------------------------- #
@sem_dm
def test_o_truncamento_do_paper_nao_muda_nenhum_veredito():
    """O que autoriza confiar em toda celula DM do manuscrito.

    Se alguma celula mudasse de veredito ao trocar o truncamento em h-1 por Newey-West, a
    escolha do estimador estaria decidindo um resultado, e isso teria que estar no texto.
    """
    d = json.loads(DM.read_text(encoding="utf-8"))
    assert d["celulas_que_mudam_total"] == 0, (
        f"{d['celulas_que_mudam_total']} de {d['celulas_total']} celulas mudam de veredito. "
        "O estimador de variancia passou a importar; documente no manuscrito.")
    assert d["celulas_total"] == 90, "a grade de comparacoes mudou de tamanho"


@sem_dm
def test_a_autocorrelacao_existe_mas_e_pequena():
    """A alegacao tinha fundamento, so nao tinha tamanho.

    Registrado porque a conclusao correta nao e "nao ha autocorrelacao" e sim "ha, e e
    pequena demais para inverter algo". As duas levam a acoes diferentes se a serie crescer.
    """
    acf = json.loads(DM.read_text(encoding="utf-8"))["acf_h1"]
    rho1 = [abs(v[0]) for v in acf.values()]
    assert max(rho1) > 0.05, "a autocorrelacao sumiu; a checagem perdeu o proposito"
    assert max(rho1) < 0.30, f"a autocorrelacao cresceu para {max(rho1):.3f}; refaca a checagem"


# --------------------------------------------------------------------------- #
# mecanismo do ganho da temperatura
# --------------------------------------------------------------------------- #
def _ganhos():
    d = pd.read_csv(TEMP)
    return {(r.modelo, r.espec): r.ganho for r in d.itertuples()}


@sem_temp
def test_o_calendario_explicito_encolhe_o_ganho_da_temperatura():
    """A predicao do manuscrito, testada.

    Se este teste falhar, a temperatura carrega informacao alem da climatologia e o
    paragrafo sobre mecanismo tem que ser reescrito, nao ajustado.
    """
    g = _ganhos()
    for kind in ("catboost", "xgboost"):
        assert g[(kind, "diffcal")] < g[(kind, "base")], (
            f"{kind}: com calendario o ganho foi {g[(kind, 'diffcal')]:.3f}, "
            f"contra {g[(kind, 'base')]:.3f} sem. A explicacao do texto nao se sustenta.")


@sem_temp
def test_no_xgboost_o_ganho_chega_a_inverter_de_sinal():
    """A forma mais forte do resultado, e a que o texto cita."""
    g = _ganhos()
    assert g[("xgboost", "base")] > 0
    assert g[("xgboost", "diffcal")] < 0


@sem_temp
def test_com_calendario_o_xgboost_se_comporta_como_o_prophet():
    """Convergencia entre dois caminhos diferentes para a mesma sazonalidade explicita.

    E o que faz a explicacao ser mecanismo e nao coincidencia: o Prophet ajusta
    sazonalidade anual por construcao, o XGBoost so passa a ter uma quando recebe o termo
    de Fourier, e os dois passam a ser prejudicados pela temperatura na mesma medida.
    """
    g = _ganhos()
    pt = RAIZ / "results" / "revisao" / "prophet_temp_vs_base.json"
    if not pt.exists():
        pytest.skip("rodada do Prophet com temperatura ausente")
    prophet = json.loads(pt.read_text(encoding="utf-8"))["modelos"]["prophet_temp"]["ganho_pp"]
    assert abs(g[("xgboost", "diffcal")] - prophet) < 0.01, (
        f"XGBoost com calendario: {g[('xgboost', 'diffcal')]:.3f}, "
        f"Prophet: {prophet:.3f}. Deixaram de convergir; o texto cita essa coincidencia.")


@sem_temp
def test_a_bancada_reproduz_o_benchmark():
    """Controle: as colunas sem temperatura tem que bater com a Tabela 1."""
    d = pd.read_csv(TEMP)
    v = json.loads((RAIZ / "paper" / "verified_numbers.json").read_text(encoding="utf-8"))
    base = {r.modelo: r.sem_T for r in d.itertuples() if r.espec == "base"}
    assert base["catboost"] == pytest.approx(v["tabela1"]["catboost"]["smape"], abs=1e-3)
    # O XGBoost depende da versao da biblioteca; ver docs/xgboost_reprodutibilidade.md.
    assert base["xgboost"] == pytest.approx(6.8324, abs=2e-3)
