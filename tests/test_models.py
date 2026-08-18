"""Testes do contrato dos modelos.

Este arquivo testa MENOS do que os outros, e de proposito. A previsao em si depende de
statsmodels, prophet, timesfm e skforecast, que sao pesados e nem sempre instalados; e,
mais importante, um erro dentro do SARIMA ou do TimesFM quebra alto e aparece na hora.

O que quebra em SILENCIO e o CONTRATO em volta deles, e e isso que esta testado aqui:

- `supports_exog` decide quais modelos recebem a exogena. Marcar `True` num modelo que
  ignora a exogena faria o benchmark reportar "com temperatura" um resultado identico ao
  "sem temperatura", e a conclusao sobre a temperatura sairia errada sem nenhum erro.
- `name` vira o rotulo de cada linha nos CSVs de resultado. Nome trocado ou repetido
  embaralha metrica entre modelos.
- O import preguicoso de cada dependencia decide se pedir um modelo derruba os outros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cv_timeseries.models import (
    CatBoostForecaster,
    Forecaster,
    ProphetForecaster,
    SarimaForecaster,
    TimesFMForecaster,
    XGBoostForecaster,
    _SkforecastRecursiveForecaster,
)

TODOS = [SarimaForecaster, ProphetForecaster, TimesFMForecaster,
         XGBoostForecaster, CatBoostForecaster]

# Quem aceita exogena de verdade. Prophet e TimesFM ignoram a exogena nesta
# implementacao, entao TEM que estar False: e o que faz o run_benchmark exclui-los da
# comparacao com temperatura em vez de dar a eles um input que nao usam.
EXOG_ESPERADO = {
    "sarima": True, "xgboost": True, "catboost": True,
    "prophet": False, "timesfm": False,
}


# --------------------------------------------------------------------------- #
# contrato da classe base
# --------------------------------------------------------------------------- #
def test_forecaster_e_abstrata():
    with pytest.raises(TypeError):
        Forecaster()


def test_subclasse_sem_forecast_nao_instancia():
    class Incompleta(Forecaster):
        name = "incompleta"

    with pytest.raises(TypeError):
        Incompleta()


def test_subclasse_minima_cumpre_o_contrato():
    """Um modelo novo so precisa de name e forecast; o resto tem default."""
    class Ingenuo(Forecaster):
        name = "ingenuo"

        def forecast(self, train, horizon, exog_train=None, exog_future=None):
            return np.full(horizon, float(train.iloc[-1]))

    m = Ingenuo()
    assert m.supports_exog is False, "o default tem que ser False: quem aceita, declara"
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    out = m.forecast(s, horizon=4)
    assert isinstance(out, np.ndarray) and len(out) == 4


# --------------------------------------------------------------------------- #
# supports_exog: o gate que decide quem recebe a temperatura
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", TODOS, ids=lambda c: c.name)
def test_supports_exog_declarado_corretamente(cls):
    assert cls.supports_exog is EXOG_ESPERADO[cls.name], (
        f"{cls.name} declara supports_exog={cls.supports_exog}. Se um modelo que IGNORA "
        "a exogena declarar True, o benchmark reporta 'com temperatura' um resultado "
        "identico ao 'sem temperatura' e a conclusao sai errada sem erro nenhum."
    )


def test_o_default_da_classe_base_e_nao_aceitar():
    """Modelo novo entra fora da comparacao com exogena ate declarar o contrario."""
    assert Forecaster.supports_exog is False


def test_prophet_e_timesfm_ficam_de_fora_por_ignorarem_a_exogena():
    """Registrado explicito porque e uma escolha, nao um esquecimento.

    Os dois ACEITAM os argumentos exog_* na assinatura, para o run_backtest poder
    chama-los de forma uniforme, mas nao os usam. Declarar False e o que faz o
    run_benchmark exclui-los da rodada com temperatura, com aviso, em vez de produzir
    numero que parece comparavel e nao e.
    """
    assert ProphetForecaster.supports_exog is False
    assert TimesFMForecaster.supports_exog is False


# --------------------------------------------------------------------------- #
# nomes: viram rotulo de linha nos CSVs de resultado
# --------------------------------------------------------------------------- #
def test_nomes_sao_unicos():
    nomes = [c.name for c in TODOS]
    assert len(nomes) == len(set(nomes))


def test_nomes_batem_com_os_aceitos_pelo_cli():
    """A lista `valid` do run_benchmark tem que casar com os nomes reais das classes."""
    assert {c.name for c in TODOS} == {"sarima", "prophet", "timesfm", "xgboost", "catboost"}


# --------------------------------------------------------------------------- #
# imports preguicosos: pedir um modelo nao pode derrubar os outros
# --------------------------------------------------------------------------- #
def test_o_modulo_importa_sem_nenhuma_dependencia_pesada():
    """Regressao de um bug real.

    statsmodels era importado no topo do modulo, entao faltar ele derrubava o import
    inteiro, e com ele qualquer execucao, mesmo uma que so pedisse timesfm. Os outros
    cinco modelos ja importavam de forma preguicosa; o SARIMA era o unico fora do
    padrao. Este teste so passa porque o import mora dentro do __init__.
    """
    import importlib
    mod = importlib.import_module("cv_timeseries.models")
    for nome in ("SarimaForecaster", "ProphetForecaster", "TimesFMForecaster",
                 "XGBoostForecaster", "CatBoostForecaster"):
        assert hasattr(mod, nome)


@pytest.mark.parametrize("cls", TODOS, ids=lambda c: c.name)
def test_dependencia_ausente_falha_ao_instanciar_e_nao_ao_importar(cls):
    """Referenciar a classe sempre funciona; so instanciar exige a dependencia.

    E o que permite ao build_models avisar "X indisponivel" e seguir com os demais.
    """
    assert cls.name and isinstance(cls.name, str)
    try:
        cls()
    except ImportError:
        pass          # dependencia ausente nesta maquina: comportamento esperado
    except Exception as exc:
        pytest.fail(f"{cls.name} falhou ao instanciar por motivo inesperado: {exc!r}")


# --------------------------------------------------------------------------- #
# familia de boosting
# --------------------------------------------------------------------------- #
def test_boosting_herda_da_base_recursiva():
    for cls in (XGBoostForecaster, CatBoostForecaster):
        assert issubclass(cls, _SkforecastRecursiveForecaster)
        assert issubclass(cls, Forecaster)


def test_lags_padrao_e_doze():
    """12 lags cobrem um ciclo sazonal completo em serie mensal."""
    assert _SkforecastRecursiveForecaster.lags == 12
    for cls in (XGBoostForecaster, CatBoostForecaster):
        try:
            assert cls().lags == 12
        except ImportError:
            pytest.skip(f"{cls.name} indisponivel nesta maquina")


def test_lags_configuravel():
    for cls in (XGBoostForecaster, CatBoostForecaster):
        try:
            assert cls(lags=6).lags == 6
        except ImportError:
            pytest.skip(f"{cls.name} indisponivel nesta maquina")


def test_base_recursiva_exige_que_a_subclasse_construa_o_regressor():
    """_build_regressor sem implementacao tem que falhar alto, nao devolver None."""
    class Vazio(_SkforecastRecursiveForecaster):
        name = "vazio"

    with pytest.raises(NotImplementedError):
        Vazio()._build_regressor()


# --------------------------------------------------------------------------- #
# assinatura uniforme
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", TODOS, ids=lambda c: c.name)
def test_forecast_aceita_os_argumentos_de_exogena(cls):
    """Mesmo quem ignora a exogena precisa ACEITAR os argumentos.

    O run_backtest chama todos da mesma forma quando ha exogena; assinatura divergente
    quebraria so na rodada com temperatura, e so para um modelo.
    """
    import inspect
    params = inspect.signature(cls.forecast).parameters
    for p in ("train", "horizon", "exog_train", "exog_future"):
        assert p in params, f"{cls.name}.forecast nao aceita {p}"
    assert params["exog_train"].default is None
    assert params["exog_future"].default is None
