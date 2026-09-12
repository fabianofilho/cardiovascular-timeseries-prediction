#!/usr/bin/env python3
"""Gera o notebook do Colab que roda o TabPFN e reconfere os demais modelos.

Por que gerar em vez de escrever a mao: o notebook precisa carregar a serie (168 pontos)
e os valores de referencia sem depender de caminho de arquivo, porque roda no Colab sem
o Drive montado. Esses dois blocos sao injetados daqui, das fontes do repositorio, pela
mesma regra do resto do projeto -- nenhum numero digitado.

O protocolo vem de kaggle/core.py, que ja e autossuficiente (nao usa skforecast) e ja
esta validado: reproduz CatBoost 6,5893 nas 103 janelas.

Fontes:
    kaggle/series_data.py          a serie, 168 meses
    paper/verified_numbers.json    sMAPE de referencia de cada modelo

Saida:
    02_experimentos/tabpfn_benchmark.ipynb

Uso:
    python scripts/revisao/build_tabpfn_notebook.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import PAPER  # noqa: E402

EXPERIMENTOS = Path(
    r"G:\.shortcut-targets-by-id\1zRZlDXqjBRVUlO0Zys2L0W69eJvgMI58\Labs"
    r"\Cardiovascular Time Series\IJF_Series_Temporais_CV\02_experimentos"
)
SERIE_PY = EXPERIMENTOS / "kaggle" / "series_data.py"
SAIDA = EXPERIMENTOS / "tabpfn_benchmark.ipynb"

# rotulos de exibicao, na ordem em que a tabela final deve sair
ORDEM_SAIDA = ["naive", "snaive", "snaive_drift", "xgboost", "catboost",
               "sarima", "prophet", "timesfm", "tabpfn", "tabpfn_ts"]


def carrega_serie() -> list[int]:
    """Le a lista SERIES do modulo do Kaggle sem importa-lo."""
    txt = SERIE_PY.read_text(encoding="utf-8")
    m = re.search(r"SERIES\s*=\s*(\[[^\]]*\])", txt, re.S)
    if not m:
        raise ValueError("SERIES nao encontrada em kaggle/series_data.py")
    serie = json.loads(m.group(1))
    if len(serie) != 168:
        raise ValueError(f"serie com {len(serie)} pontos, esperado 168")
    return serie


def carrega_referencia() -> dict:
    v = json.loads((PAPER / "verified_numbers.json").read_text(encoding="utf-8"))
    ref = {m: d["smape"] for m, d in v["tabela1"].items()}
    return {"smape": ref, "backtest": v["backtest"]}


# --------------------------------------------------------------------- celulas

def fonte(txt: str) -> list[str]:
    """Quebra em linhas MANTENDO o \\n de cada uma.

    O nbformat manda que source seja o texto ja quebrado, e o Jupyter remonta com
    ''.join(source), sem separador. Se as linhas vierem sem o \\n final, a celula
    inteira colapsa numa linha so e o notebook nao roda. Foi exatamente esse o defeito
    da primeira versao deste gerador.
    """
    return txt.strip("\n").splitlines(keepends=True)


def md(txt):
    return {"cell_type": "markdown", "metadata": {}, "source": fonte(txt)}


def code(txt):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": fonte(txt)}


CEL_INSTALA = '''
# Instalacao. Leva alguns minutos. Se o Colab pedir para reiniciar o ambiente, reinicie
# e comece de novo desta celula -- nao pule para as seguintes sem reiniciar.
# tabpfn-client 0.3.0 e a versao com suporte ao TabPFN-3 (ate 1M linhas x 200 atributos).
!pip -q install --upgrade tabpfn-client
!pip -q install -q xgboost catboost prophet statsmodels
# TabPFN-TS so e usado na celula opcional do enquadramento B; se falhar, siga assim mesmo.
!pip -q install tabpfn-time-series 2>/dev/null || echo "tabpfn-time-series indisponivel (celula B sera pulada)"
print("instalacao terminada")
'''

CEL_IMPORTS = '''
# Imports numa celula unica, e a chave da PriorLabs.
import json, os, re, time, warnings
from getpass import getpass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# A chave vem dos Secrets do Colab. O segredo se chama PRIOR_LABS_TOKEN; a biblioteca
# le a variavel de ambiente TABPFN_TOKEN, entao um alimenta o outro aqui.
# tabpfn_client.fit() levanta erro sem token e nunca abre prompt sozinho, entao o token
# tem de estar definido antes de qualquer chamada.
NOMES_SEGREDO = ["PRIOR_LABS_TOKEN", "TABPFN_TOKEN"]

TOKEN, origem = os.environ.get("TABPFN_TOKEN"), "variavel de ambiente"
if not TOKEN:
    try:
        from google.colab import userdata
        for _nome in NOMES_SEGREDO:
            try:
                TOKEN = userdata.get(_nome)
            except Exception:
                TOKEN = None
            if TOKEN:
                origem = f"Secrets do Colab ({_nome})"
                break
    except ImportError:
        pass
if not TOKEN:
    TOKEN = getpass("Token da PriorLabs (platform.priorlabs.ai/account/api-keys): ")
    origem = "digitado agora"
os.environ["TABPFN_TOKEN"] = TOKEN.strip()
print(f"token definido, {len(os.environ['TABPFN_TOKEN'])} caracteres, de {origem}")
'''

CEL_PROTOCOLO = '''
# Protocolo do artigo, portado de kaggle/core.py. Janela expansiva, minimo 60 meses de
# treino, horizonte 6, origem rolante: 168 - 60 - 6 + 1 = 103 janelas.
HORIZONTE = 6
MIN_TRAIN = 60
LAGS = 12

def rolling_origin_splits(serie, horizonte=HORIZONTE, min_train=MIN_TRAIN):
    n = len(serie)
    for fim in range(min_train, n - horizonte + 1):
        yield serie[:fim], serie[fim:fim + horizonte], fim

def smape(a, b, eps=1e-8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = (np.abs(a) + np.abs(b) + eps) / 2.0
    return float(np.mean(np.abs(a - b) / den) * 100.0)

def make_supervised(y, lags):
    """X[t] = [y_{t-1}, ..., y_{t-lags}], igual ao ForecasterRecursive do skforecast."""
    linhas, alvo = [], []
    for t in range(lags, len(y)):
        linhas.append([y[t - k] for k in range(1, lags + 1)])
        alvo.append(y[t])
    return np.asarray(linhas, float), np.asarray(alvo, float)

def avalia(prever, nome="", verboso=True):
    """prever(treino_np, horizonte, indice_datas) -> array de tamanho horizonte."""
    yt, yp = [], []
    t0 = time.time()
    for treino, teste, fim in rolling_origin_splits(SERIE):
        p = np.asarray(prever(np.asarray(treino, float), HORIZONTE, DATAS[:fim]), float)
        if p.shape != (HORIZONTE,) or not np.all(np.isfinite(p)):
            raise ValueError(f"{nome}: previsao invalida na janela que termina em {fim}")
        yt.append(teste); yp.append(p)
    if nome:
        ALVOS[nome] = np.concatenate(yt)
    s = smape(np.concatenate(yt), np.concatenate(yp))
    if verboso:
        print(f"  {nome:14s} sMAPE {s:.6f}   ({len(yt)} janelas, {time.time() - t0:.0f}s)")
    return s

# GEOMETRIA guarda, por modelo, a forma exata do que cada janela viu: onde ela termina,
# quantos meses de treino, quantas linhas supervisionadas e quantos lags. E o que permite
# provar, no fim, que o TabPFN recebeu a MESMA amostra que os boosters, em vez de a gente
# acreditar que recebeu.
GEOMETRIA = {}

# ALVOS guarda o vetor de valores reais contra o qual cada modelo foi pontuado. Se dois
# modelos foram avaliados no mesmo split, estes vetores sao identicos byte a byte.
ALVOS = {}

def preditor_recursivo(constroi_modelo, lags=LAGS, ajusta=None, preve=None, nome=None):
    """Fecha um regressor sklearn-compativel no protocolo recursivo do artigo.

    `ajusta` e `preve` existem para o TabPFN, que precisa de limitador de taxa e
    retentativa em volta de cada chamada de rede. Eles NAO mudam o protocolo: envolvem
    a mesma chamada. O objetivo e que exista um unico caminho recursivo no notebook,
    para que boosters e TabPFN nao possam divergir sem alguem notar.
    """
    ajusta = ajusta or (lambda m, X, y: m.fit(X, y))
    preve = preve or (lambda m, x: m.predict(x))

    def prever(treino, horizonte, datas):
        L = max(1, min(lags, len(treino) - 1))
        X, y = make_supervised(treino, L)
        if nome:
            GEOMETRIA.setdefault(nome, []).append(
                (len(datas), len(treino), int(X.shape[0]), int(X.shape[1])))
        m = constroi_modelo()
        ajusta(m, X, y)
        hist = list(map(float, treino))
        saida = []
        for _ in range(horizonte):
            x = np.asarray([[hist[-k] for k in range(1, L + 1)]], float)
            v = float(preve(m, x)[0])
            saida.append(v)
            hist.append(v)
        return np.asarray(saida)
    return prever

RESULTADOS = {}
print(f"{sum(1 for _ in rolling_origin_splits(SERIE))} janelas, "
      f"horizonte {HORIZONTE}, minimo de treino {MIN_TRAIN}")
'''

CEL_INGENUAS = '''
# Portao de entrada. As tres referencias ingenuas sao deterministicas: se elas nao
# baterem com o artigo ate a sexta casa, o protocolo deste notebook nao e o mesmo do
# artigo e nada abaixo vale.
def naive(treino, horizonte, datas):
    return np.full(horizonte, float(treino[-1]))

def snaive(treino, horizonte, datas, m=12):
    if len(treino) < m:
        return np.full(horizonte, float(treino[-1]))
    return np.array([float(treino[-m + i % m]) for i in range(horizonte)])

def snaive_drift(treino, horizonte, datas, m=12):
    base = snaive(treino, horizonte, datas, m)
    if len(treino) <= m + 1:
        return base
    drift = float(treino[-1]) - float(treino[-m - 1])
    return base + drift * np.arange(1, horizonte + 1) / m

for nome, fn in (("naive", naive), ("snaive", snaive), ("snaive_drift", snaive_drift)):
    RESULTADOS[nome] = avalia(fn, nome)

falhas = [n for n in ("naive", "snaive", "snaive_drift")
          if abs(RESULTADOS[n] - REFERENCIA["smape"][n]) > 1e-6]
if falhas:
    raise SystemExit(f"PROTOCOLO DIVERGENTE em {falhas}. Pare aqui: o resto nao e comparavel.")
print("\\nProtocolo confere com o artigo nas tres ingenuas. Pode seguir.")
'''

CEL_BOOSTING = '''
# XGBoost e CatBoost com os hiperparametros do artigo (fixos, escolhidos a priori).
# A versao da biblioteca e impressa junto: o XGBoost e o unico modelo do estudo que nao
# reproduz entre versoes, e sem a versao o numero nao diz nada.
import xgboost, catboost
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
print("xgboost", xgboost.__version__, "| catboost", catboost.__version__)

RESULTADOS["xgboost"] = avalia(preditor_recursivo(lambda: XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
), nome="xgboost"), "xgboost")

RESULTADOS["catboost"] = avalia(preditor_recursivo(lambda: CatBoostRegressor(
    iterations=300, depth=4, learning_rate=0.05, random_seed=42,
    verbose=False, allow_writing_files=False,
), nome="catboost"), "catboost")
'''

CEL_CLASSICOS = '''
# SARIMA (1,1,1)(0,1,1,12) e Prophet com sazonalidade anual apenas, como no artigo.
# Esta celula demora alguns minutos: sao 103 ajustes por modelo.
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

def sarima(treino, horizonte, datas):
    # enforce_stationarity/invertibility=True e maxiter=200 nao sao detalhe: sao o que
    # src/cv_timeseries/models.py usa. Com os dois em False o modelo fica mais livre,
    # acha um otimo diferente e devolve sMAPE ~0,22 pp menor -- numero bonito e
    # incomparavel com a Tabela 1.
    s = pd.Series(treino, index=datas)
    ajuste = SARIMAX(s, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                     enforce_stationarity=True,
                     enforce_invertibility=True).fit(disp=False, maxiter=200)
    return ajuste.forecast(steps=horizonte).to_numpy(dtype=float)

def prophet(treino, horizonte, datas):
    df = pd.DataFrame({"ds": datas, "y": treino})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False).fit(df)
    fut = m.make_future_dataframe(periods=horizonte, freq="MS")
    return m.predict(fut).tail(horizonte)["yhat"].to_numpy(dtype=float)

RESULTADOS["sarima"] = avalia(sarima, "sarima")
RESULTADOS["prophet"] = avalia(prophet, "prophet")
'''

CEL_TIMESFM = '''
# OPCIONAL e pesada: TimesFM 2.5, zero-shot. Pede GPU e alguns GB de download.
# Pule se so quiser conferir o TabPFN; a referencia do artigo e 4,832521.
RODAR_TIMESFM = False

if RODAR_TIMESFM:
    !pip -q install timesfm
    import timesfm
    modelo_tfm = timesfm.TimesFm_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    modelo_tfm.compile(timesfm.ForecastConfig(
        max_context=512, max_horizon=HORIZONTE, normalize_inputs=True))

    def timesfm_prever(treino, horizonte, datas):
        pontual, _ = modelo_tfm.forecast(horizon=horizonte, inputs=[treino])
        return np.asarray(pontual[0][:horizonte], float)

    RESULTADOS["timesfm"] = avalia(timesfm_prever, "timesfm")
else:
    print("TimesFM pulado (RODAR_TIMESFM = False)")
'''

CEL_TABPFN = '''
#@title A. TabPFN tabular - depois do teste, TROQUE PARA completo { display-mode: "form" }
#@markdown O padrao e `teste`, que gasta 5 ajustes e prova a chave em meio minuto.
#@markdown **Para o resultado que vai para o artigo, troque para `completo` e rode de
#@markdown novo.** Leva cerca de uma hora e retoma sozinho se cair.
MODO = "teste (5 janelas)" #@param ["teste (5 janelas)", "completo (103 janelas)", "nao rodar"]
THINKING = False #@param {type:"boolean"}
#@markdown ---

# O revisor pediu o TabPFN como competidor de XGBoost e CatBoost, entao ele entra no
# MESMO protocolo deles: lags 1-12, recursivo, janela expansiva. E trocar o regressor,
# nada mais, para que a comparacao seja do modelo e nao da featurizacao.
#
# Custo do modo completo: 103 ajustes + 618 predicoes na API. Limites da PriorLabs sao
# 60 fits/min (1.000/h) e 60 predicoes/min (1.500/h), entao cabe numa hora com pausa.
#
# THINKING melhora ate 15%, mas o limite cai para 10 fits/min e 30/h: mais de tres horas
# e meia para as 103 janelas. So ligue sabendo disso.

CHECKPOINT = "tabpfn_previsoes.csv"
CONFIRMO_RODAR = MODO != "nao rodar"
LIMITE_JANELAS = 5 if MODO.startswith("teste") else None

n_janelas = sum(1 for _ in rolling_origin_splits(SERIE))
if LIMITE_JANELAS:
    n_janelas = min(n_janelas, LIMITE_JANELAS)
print(f"modo: {MODO}")
print(f"custo: {n_janelas} ajustes e {n_janelas * HORIZONTE} predicoes na API.")
if THINKING:
    print("THINKING ligado: 30 ajustes por hora, estime ~3,5h no modo completo.")

class LimitadorDeTaxa:
    """Espera o minimo para nao passar de `por_minuto` chamadas por minuto."""
    def __init__(self, por_minuto):
        self.intervalo = 60.0 / por_minuto
        self.ultima = 0.0
    def espera(self):
        agora = time.time()
        atraso = self.intervalo - (agora - self.ultima)
        if atraso > 0:
            time.sleep(atraso)
        self.ultima = time.time()

def com_retentativa(fn, limitador, tentativas=5):
    """Repete em HTTP 429 respeitando o Retry-After em vez de tentar as cegas."""
    for k in range(tentativas):
        limitador.espera()
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "Too Many Requests" not in msg.lower():
                raise
            espera = 60.0
            m = re.search(r"[Rr]etry-?[Aa]fter[^0-9]*(\\d+)", msg)
            if m:
                espera = float(m.group(1))
            print(f"    429; esperando {espera:.0f}s (tentativa {k + 1}/{tentativas})")
            time.sleep(espera + 1)
    raise RuntimeError("limite de taxa persistente apos varias tentativas")

def constroi_tabpfn(TabPFNRegressor):
    # thinking_mode/effort/metric so existem na 0.3.0; se a versao instalada for antiga,
    # cai para o construtor simples em vez de quebrar no meio da rodada.
    if THINKING:
        try:
            return TabPFNRegressor(thinking_mode=True, thinking_effort="high",
                                   thinking_metric="rmse")
        except TypeError:
            print("    aviso: esta versao do cliente nao aceita thinking_mode")
    return TabPFNRegressor()

if CONFIRMO_RODAR:
    # O import fica aqui dentro de proposito: com o modo em "nao rodar" a celula roda e
    # mostra a estimativa de custo mesmo sem o pacote instalado.
    import tabpfn_client
    from tabpfn_client import TabPFNRegressor

    # Alem da variavel de ambiente, entrega o token pela via oficial do cliente. As duas
    # formas existem e nem toda versao le a variavel.
    try:
        tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    except Exception as e:
        print(f"aviso ao registrar o token: {type(e).__name__}: {e}")

    # No modo teste o checkpoint e ignorado: ele guarda janelas da rodada completa e
    # misturar as duas daria um sMAPE que nao e de nenhuma das duas.
    feitas = {}
    if LIMITE_JANELAS is None and os.path.exists(CHECKPOINT):
        ck = pd.read_csv(CHECKPOINT)
        for fim, g in ck.groupby("fim"):
            feitas[int(fim)] = g.sort_values("h").y_pred.to_numpy(float)
        print(f"retomando: {len(feitas)} janelas ja no checkpoint")

    lim_fit = LimitadorDeTaxa(50)      # folga sobre o limite de 60/min
    lim_pred = LimitadorDeTaxa(50)
    linhas_ck = []
    yt_all, yp_all = [], []
    t0 = time.time()

    for i, (treino, teste, fim) in enumerate(rolling_origin_splits(SERIE), 1):
        if LIMITE_JANELAS and i > LIMITE_JANELAS:
            break
        if fim in feitas:
            p = feitas[fim]
        else:
            # MESMA funcao que roda XGBoost e CatBoost. O limitador de taxa entra pelos
            # ganchos `ajusta`/`preve`, envolvendo a chamada sem tocar no protocolo.
            prever_tabpfn = preditor_recursivo(
                lambda: constroi_tabpfn(TabPFNRegressor),
                ajusta=lambda m, X, y: com_retentativa(lambda: m.fit(X, y), lim_fit),
                preve=lambda m, x: com_retentativa(lambda: m.predict(x), lim_pred),
                nome="tabpfn")
            p = prever_tabpfn(np.asarray(treino, float), HORIZONTE, DATAS[:fim])
            if LIMITE_JANELAS is None:
                for h, v in enumerate(p, 1):
                    linhas_ck.append({"fim": fim, "h": h, "y_pred": v})
                existia = os.path.exists(CHECKPOINT)
                pd.DataFrame(linhas_ck).to_csv(
                    CHECKPOINT, index=False, mode="a" if existia else "w",
                    header=not existia)
                linhas_ck = []
        yt_all.append(teste); yp_all.append(p)
        if i % 10 == 0 or LIMITE_JANELAS:
            print(f"  janela {i}/{n_janelas}  ({time.time() - t0:.0f}s)")

    ALVOS["tabpfn"] = np.concatenate(yt_all)
    s = smape(np.concatenate(yt_all), np.concatenate(yp_all))
    if LIMITE_JANELAS:
        # O sMAPE de 5 janelas nao e comparavel com o das 103 do artigo, entao ele NAO
        # entra em RESULTADOS: entrar ali seria contrabandear um numero de outro
        # experimento para dentro da tabela final.
        print("")
        print("=" * 68)
        print(f"  TESTE OK: {n_janelas} janelas em {time.time() - t0:.0f}s, "
              f"sMAPE parcial {s:.4f}")
        print("  A chave e a API funcionam. Este numero NAO vai para o artigo:")
        print("  5 janelas nao se comparam com as 103 dos outros modelos.")
        print("")
        print("  >>> Agora troque MODO para 'completo (103 janelas)' aqui em cima")
        print("  >>> e rode ESTA celula de novo. Depois rode a ultima celula.")
        print("=" * 68)
    else:
        RESULTADOS["tabpfn"] = s
        print(f"\\n  tabpfn         sMAPE {s:.6f}"
              f"   ({n_janelas} janelas, {time.time() - t0:.0f}s)")
else:
    print("")
    print("=" * 68)
    print("  NAO RODOU. O TabPFN e o motivo deste notebook existir.")
    print("  No formulario acima, troque MODO de 'nao rodar' para")
    print("  'teste (5 janelas)' e execute esta celula de novo.")
    print("=" * 68)
'''

CEL_PROVA_SPLIT = '''
# Prova de que o TabPFN viu a MESMA amostra e o MESMO split que os boosters.
# Nao e conferencia de olho no codigo: compara o que cada janela recebeu de fato.
if "tabpfn" not in GEOMETRIA:
    print("TabPFN ainda nao rodou; nada a comparar. Rode a celula A primeiro.")
else:
    ref_nome = "catboost" if "catboost" in GEOMETRIA else "xgboost"
    ref = {g[0]: g for g in GEOMETRIA[ref_nome]}
    alvo = {g[0]: g for g in GEOMETRIA["tabpfn"]}

    # 1. cada janela do TabPFN existe no booster e tem exatamente a mesma forma
    divergentes = [(f, alvo[f], ref.get(f)) for f in sorted(alvo)
                   if ref.get(f) != alvo[f]]
    if divergentes:
        for f, a, b in divergentes[:5]:
            print(f"  janela {f}: tabpfn {a}  !=  {ref_nome} {b}")
        raise SystemExit("DIVERGENCIA de amostra entre TabPFN e boosters")

    # 2. cobertura: quantas das 103 janelas o TabPFN cobriu
    todas = sorted(ref)
    completo = sorted(alvo) == todas

    # 3. o vetor de valores reais e identico byte a byte
    mesmo_alvo = None
    if "tabpfn" in ALVOS and ref_nome in ALVOS and completo:
        mesmo_alvo = bool(np.array_equal(ALVOS["tabpfn"], ALVOS[ref_nome]))

    linhas_max = max(g[2] for g in GEOMETRIA["tabpfn"])
    print(f"janelas do TabPFN: {len(alvo)} de {len(todas)}"
          + ("  (COMPLETO)" if completo else "  (PARCIAL - modo teste)"))
    print(f"origens: de {todas[0]} a {todas[-1]} meses de treino, passo 1, expansiva")
    print(f"forma identica ao {ref_nome} em todas as janelas comparadas: SIM")
    print(f"maior matriz entregue ao TabPFN: {linhas_max} linhas x "
          f"{GEOMETRIA['tabpfn'][0][3]} atributos")
    print("limite do TabPFN-3: 1.000.000 linhas x 200 atributos -> "
          "nenhuma amostragem, nenhum truncamento")
    if mesmo_alvo is not None:
        print(f"vetor de valores reais identico ao do {ref_nome}: "
              + ("SIM" if mesmo_alvo else "NAO")
              + f"  ({len(ALVOS['tabpfn'])} previsoes pontuadas)")
        if not mesmo_alvo:
            raise SystemExit("os dois modelos foram pontuados contra alvos diferentes")
    if not completo:
        print("")
        print("Parcial porque a celula A rodou no modo teste. Rode no modo completo")
        print("para que esta prova cubra as 103 janelas.")
'''


CEL_OPTUNA = '''
#@title Optuna SEM vazamento - busca honesta nos boosters { display-mode: "form" }
#@markdown Ajusta hiperparametros num backtest interno restrito aos **60 primeiros
#@markdown meses**, que sao treino de todas as 103 janelas do benchmark. Nenhuma data
#@markdown pontuada e vista pela busca. Depois avalia UMA vez no benchmark completo.
#@markdown Custo alto e variavel: conte com horas para 40 trials nos dois modelos.
RODAR_OPTUNA = False #@param {type:"boolean"}
TRIALS = 40 #@param {type:"integer"}
#@markdown ---

# Por que este desenho e honesto:
#
#   SERIE[:60]  -> espaco de busca. Todas as 103 janelas do benchmark tem no minimo 60
#                  meses de treino, entao estes 60 meses sao treino em TODAS elas.
#   SERIE[60:]  -> nunca entra na busca. E o periodo pontuado.
#
# A busca roda um backtest proprio dentro dos 60 meses (min_train 36, horizonte 6, as
# mesmas origens rolantes). Os hiperparametros vencedores saem dali e so entao sao
# avaliados no benchmark completo, uma unica vez.
#
# O paper tambem reporta um TETO COM VAZAMENTO, que otimiza direto no sMAPE das 103
# janelas. Aquele numero nao e desempenho, e limite superior: responde "e se voces
# tivessem ajustado melhor?" com "mesmo trapaceando nao chega". Ele NAO e reproduzido
# aqui, de proposito -- este notebook so produz numeros reportaveis.

DEV_MESES = 60

if RODAR_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    assert DEV_MESES == MIN_TRAIN, (
        f"o split honesto so vale se DEV_MESES == MIN_TRAIN ({MIN_TRAIN})")
    serie_dev = SERIE[:DEV_MESES]

    def avalia_em(serie, prever, horizonte=HORIZONTE, min_train=MIN_TRAIN):
        yt, yp = [], []
        for treino, teste, fim in rolling_origin_splits(serie, horizonte, min_train):
            try:
                pr = prever(np.asarray(treino, float), horizonte, DATAS[:fim])
            except Exception:
                return float("inf")
            if not np.all(np.isfinite(pr)):
                return float("inf")
            yt.append(teste); yp.append(pr)
        return smape(np.concatenate(yt), np.concatenate(yp)) if yt else float("inf")

    def espaco(trial, kind):
        if kind == "xgboost":
            return dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 1500, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                random_state=42, n_jobs=-1)
        return dict(
            iterations=trial.suggest_int("iterations", 100, 1500, step=50),
            depth=trial.suggest_int("depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            random_seed=42, verbose=False, allow_writing_files=False)

    def constroi(kind, params):
        if kind == "xgboost":
            return lambda: XGBRegressor(**params)
        return lambda: CatBoostRegressor(**params)

    OPTUNA = {}
    for kind in ("xgboost", "catboost"):
        t0 = time.time()

        def objetivo(trial, kind=kind):
            params = espaco(trial, kind)
            lags = trial.suggest_int("lags", 6, 24)
            # a busca so enxerga serie_dev; min_train 36 porque sao so 60 meses
            return avalia_em(serie_dev,
                             preditor_recursivo(constroi(kind, params), lags=lags),
                             min_train=36)

        estudo = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=20260817))
        estudo.optimize(objetivo, n_trials=TRIALS, show_progress_bar=False)

        melhor = dict(estudo.best_params)
        lags = melhor.pop("lags")
        params = espaco(optuna.trial.FixedTrial({**melhor, "lags": lags}), kind)
        # avaliacao final: benchmark completo, uma vez, com os hiperparametros da busca
        final = avalia_em(SERIE, preditor_recursivo(constroi(kind, params), lags=lags))
        base = RESULTADOS.get(kind)
        OPTUNA[kind] = {"objetivo_dev": estudo.best_value, "smape_benchmark": final,
                        "lags": lags, "trials": TRIALS,
                        "segundos": round(time.time() - t0)}
        msg = (f"  {kind:9s} dev {estudo.best_value:.4f} -> benchmark {final:.6f}"
               f"   lags {lags}   ({TRIALS} trials, {time.time() - t0:.0f}s)")
        if base is not None:
            msg += f"   ganho {base - final:+.3f} pp sobre o nao-ajustado"
        print(msg)

    sn = REFERENCIA["smape"].get("snaive")
    if sn:
        print("")
        print(f"naive sazonal: {sn:.4f}. Um booster ajustado so muda a conclusao do")
        print("artigo se ficar abaixo disso.")
else:
    OPTUNA = {}
    print("Optuna desligado. Ligue RODAR_OPTUNA acima.")
    print("Tempo: muito variavel, porque o custo de cada trial depende do numero de")
    print("arvores e de lags que o sampler propoe. Medido aqui: 3 trials levaram 60s")
    print("no XGBoost e 551s no CatBoost. Conte com algumas horas para 40 trials nos")
    print("dois, e prefira rodar com a sessao em segundo plano.")
'''

CEL_TABPFN_TS = '''
#@title B. TabPFN-TS, opcional { display-mode: "form" }
RODAR_TABPFN_TS = False #@param {type:"boolean"}
#@markdown ---

# Featurizacao propria (indice corrido, calendario, sazonalidade automatica) em vez das
# lags do artigo. Nao e o que o parecer pediu, mas responde a pergunta vizinha: a
# diferenca, se houver, vem do modelo ou da forma de apresentar a serie a ele? E o
# analogo do TimesFM, modelo de fundacao aplicado a serie, e custa so 103 chamadas.

if RODAR_TABPFN_TS:
    from tabpfn_time_series import (TimeSeriesDataFrame, FeatureTransformer,
                                    TabPFNTimeSeriesPredictor, TabPFNMode)
    from tabpfn_time_series.features import (RunningIndexFeature, CalendarFeature,
                                             AutoSeasonalFeature)
    atributos = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    preditor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)

    n_janelas = sum(1 for _ in rolling_origin_splits(SERIE))
    yt_all, yp_all = [], []
    for i, (treino, teste, fim) in enumerate(rolling_origin_splits(SERIE), 1):
        tr = TimeSeriesDataFrame(pd.DataFrame({
            "item_id": "sp", "timestamp": DATAS[:fim], "target": treino}))
        fut = TimeSeriesDataFrame(pd.DataFrame({
            "item_id": "sp", "timestamp": DATAS[fim:fim + HORIZONTE],
            "target": np.nan}))
        tr, fut = FeatureTransformer(atributos).transform(tr, fut)
        pred = preditor.predict(tr, fut)
        yt_all.append(teste)
        yp_all.append(np.asarray(pred["target"].to_numpy(), float)[:HORIZONTE])
        if i % 10 == 0:
            print(f"  janela {i}/{n_janelas}")

    RESULTADOS["tabpfn_ts"] = smape(np.concatenate(yt_all), np.concatenate(yp_all))
    print(f"  tabpfn_ts      sMAPE {RESULTADOS['tabpfn_ts']:.6f}")
else:
    print("nao rodou: mude RODAR_TABPFN_TS para True")
'''

CEL_FINAL = '''
# Tabela final: o que este notebook obteve contra o que o artigo reporta.
# "bate" = diferenca abaixo de 0,001 pp. "desvia" = acima disso, e ai o motivo importa.
linhas = []
for nome in ORDEM_SAIDA:
    obtido = RESULTADOS.get(nome)
    ref = REFERENCIA["smape"].get(nome)
    if obtido is None:
        continue
    if ref is None:
        veredito, dif = "novo", None
    else:
        dif = obtido - ref
        veredito = "bate" if abs(dif) < 1e-3 else "desvia"
    linhas.append({"modelo": nome, "smape_obtido": round(obtido, 6),
                   "smape_artigo": None if ref is None else round(ref, 6),
                   "diferenca_pp": None if dif is None else round(dif, 6),
                   "veredito": veredito})

tabela = pd.DataFrame(linhas)
print(tabela.to_string(index=False))

if "tabpfn" not in RESULTADOS and "tabpfn_ts" not in RESULTADOS:
    print("")
    print("=" * 68)
    print("  ATENCAO: este arquivo NAO tem nenhum resultado de TabPFN.")
    print("  Volte a celula 'A. TabPFN tabular', troque MODO para")
    print("  'completo (103 janelas)', rode aquela celula e depois esta.")
    print("=" * 68)

# As versoes entram no arquivo porque sem elas um "desvia" nao tem como ser explicado:
# o XGBoost muda de resultado entre versoes, e o Prophet muda um pouco tambem.
versoes = {}
for _nome in ("xgboost", "catboost", "prophet", "statsmodels", "tabpfn_client",
              "numpy", "pandas"):
    try:
        versoes[_nome] = __import__(_nome).__version__
    except Exception:
        versoes[_nome] = None

# A prova de split viaja junto do resultado: quem receber este arquivo consegue saber,
# sem rodar nada, que o TabPFN foi avaliado na mesma amostra que os boosters.
prova = None
if "tabpfn" in GEOMETRIA:
    _ref = "catboost" if "catboost" in GEOMETRIA else "xgboost"
    _r = {g[0]: g for g in GEOMETRIA.get(_ref, [])}
    _a = {g[0]: g for g in GEOMETRIA["tabpfn"]}
    prova = {
        "referencia": _ref,
        "janelas_tabpfn": len(_a),
        "janelas_referencia": len(_r),
        "split_completo": sorted(_a) == sorted(_r) and len(_r) > 0,
        "forma_identica": all(_r.get(f) == _a[f] for f in _a),
        "alvo_identico": bool(
            "tabpfn" in ALVOS and _ref in ALVOS
            and len(ALVOS["tabpfn"]) == len(ALVOS[_ref])
            and np.array_equal(ALVOS["tabpfn"], ALVOS[_ref])),
        "maior_matriz_treino": [max(g[2] for g in GEOMETRIA["tabpfn"]),
                                GEOMETRIA["tabpfn"][0][3]],
    }

saida = {
    "_meta": {
        "gerado_por": "tabpfn_benchmark.ipynb",
        "protocolo": "103 janelas, horizonte 6, minimo de treino 60, janela expansiva",
        "thinking_mode": bool(globals().get("THINKING", False)),
        "versoes": versoes,
        "modelos_ausentes": [m for m in ORDEM_SAIDA if m not in RESULTADOS],
        "prova_split": prova,
        "optuna_honesto": globals().get("OPTUNA") or None,
    },
    "resultados": linhas,
}
with open("tabpfn_resultados.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)
print("\\nEscrito tabpfn_resultados.json -- baixe e me devolva este arquivo.")
try:
    from google.colab import files
    files.download("tabpfn_resultados.json")
except Exception:
    pass
'''


def valida(nb) -> None:
    """Remonta cada celula COMO O JUPYTER remonta e compila o Python resultante.

    A checagem tem de partir de ''.join(source), nao de '\\n'.join(source): foi por
    testar a segunda forma que a primeira versao deste gerador passou num notebook em
    que toda celula colapsava numa linha unica no Colab.
    """
    for i, c in enumerate(nb["cells"]):
        texto = "".join(c["source"])
        if c["cell_type"] != "code":
            continue
        if len(c["source"]) > 1 and "\n" not in texto:
            raise ValueError(f"celula {i}: linhas sem quebra, colapsaria no Jupyter")
        # Linhas de shell (!pip) nao sao Python; viram no-op so para o compile. A
        # indentacao tem de ser preservada, senao um "!pip" dentro de um if vira erro
        # de bloco -- no IPython ele funciona, virando get_ipython().system(...).
        limpo = "\n".join(
            (l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith("!") else l
            for l in texto.split("\n"))
        try:
            compile(limpo, f"<celula {i}>", "exec")
        except SyntaxError as e:
            raise ValueError(f"celula {i}: {e}") from e
    print(f"  validadas {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} "
          "celulas de codigo")


def main() -> int:
    serie = carrega_serie()
    ref = carrega_referencia()

    cel_dados = (
        "# Serie e referencias, injetadas pelo gerador a partir do repositorio.\n"
        "# Obitos cardiovasculares mensais no estado de Sao Paulo, jan/2010 a dez/2023.\n"
        f"SERIE = {serie}\n\n"
        f"REFERENCIA = {json.dumps(ref, ensure_ascii=False)}\n\n"
        f"ORDEM_SAIDA = {ORDEM_SAIDA}\n\n"
        'DATAS = pd.date_range("2010-01-01", periods=len(SERIE), freq="MS")\n'
        'assert len(SERIE) == 168, len(SERIE)\n'
        'print(f"serie com {len(SERIE)} meses, de {DATAS[0]:%Y-%m} a {DATAS[-1]:%Y-%m}")\n'
        'print("referencias:", {k: round(v, 4) for k, v in REFERENCIA["smape"].items()})'
    )

    celulas = [
        md("""
# TabPFN no benchmark de mortalidade cardiovascular

Este notebook fecha o ultimo comentario do parecer: *"o melhor do tabular precisa entrar
pra ser justa a comparacao"*. Roda o **TabPFN-3** no mesmo protocolo dos demais modelos e,
de quebra, reconfere os outros para provar que o protocolo aqui e o mesmo do artigo.

**Antes de rodar:** a chave da PriorLabs sai dos *Secrets* do Colab, do segredo
`PRIOR_LABS_TOKEN` (icone da chave na barra lateral, com o acesso ao notebook ligado).
Se ele nao existir, a celula pede a chave por prompt.

**Ordem das celulas.** As tres primeiras preparam o ambiente. A celula das ingenuas e um
portao: se ela falhar, pare, porque o protocolo divergiu e nada abaixo e comparavel.

**A celula que importa e a "A. TabPFN tabular".** Ela ja vem no modo `teste (5 janelas)`,
que gasta quase nada e prova que a chave e a API funcionam. **Depois que o teste passar,
troque o menu para `completo (103 janelas)` e rode a celula de novo** -- so essa rodada
vale para o artigo. Sem ela o arquivo final sai sem TabPFN, que e justamente o que o
parecer pediu.
"""),
        code(CEL_INSTALA),
        code(CEL_IMPORTS),
        code(cel_dados),
        code(CEL_PROTOCOLO),
        md("## Portao: as tres referencias ingenuas"),
        code(CEL_INGENUAS),
        md("## Modelos do artigo, reconferidos"),
        code(CEL_BOOSTING),
        code(CEL_CLASSICOS),
        code(CEL_TIMESFM),
        md("## TabPFN"),
        code(CEL_TABPFN),
        code(CEL_TABPFN_TS),
        md("""
## A amostra e o split sao os mesmos?

A celula abaixo nao acredita no codigo: ela compara o que cada janela **entregou de
fato** ao TabPFN e ao booster -- onde a janela termina, quantos meses de treino, quantas
linhas supervisionadas, quantos lags -- e confere se os dois foram pontuados contra o
mesmo vetor de valores reais. Se divergir em qualquer janela, para com erro.
"""),
        code(CEL_PROVA_SPLIT),
        md("""
## Optuna sem vazamento

Busca honesta de hiperparametros: o espaco de busca e um backtest interno restrito aos
60 primeiros meses, que sao treino de **todas** as 103 janelas do benchmark. Nenhuma data
pontuada entra na busca. O teto com vazamento que o artigo tambem reporta nao e
reproduzido aqui de proposito -- ele e um limite superior, nao um resultado.
"""),
        code(CEL_OPTUNA),
        md("## Resultado"),
        code(CEL_FINAL),
    ]

    nb = {
        "cells": celulas,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    valida(nb)
    SAIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  notebook  {SAIDA.name}  ({len(celulas)} celulas, "
          f"{SAIDA.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
