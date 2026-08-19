#!/usr/bin/env python3
"""Calibracao dos intervalos de previsao do SARIMA e do Prophet.

O manuscrito diz que nenhum dos modelos produz intervalo de previsao e chama a ausencia de
calibracao de lacuna mais consequente do trabalho. A primeira metade da frase esta errada:
SARIMA e Prophet produzem intervalo nativo, so nao estavam sendo pedidos. Este script pede.

O que se mede, a 95% nominal:

  PICP  cobertura empirica, fracao de observacoes dentro do intervalo. Alvo 0,95.
  MPIW  largura media do intervalo, em obitos por mes. Sozinha nao diz nada: intervalo
        largo cobre bem por construcao. Serve para ler o PICP.
  IS    interval score de Gneiting e Raftery. Combina as duas: soma a largura e adiciona
        penalidade proporcional a distancia quando a observacao cai fora. Menor e melhor.

Mesmo protocolo do benchmark: 103 janelas de origem rolante, horizonte 6, treino minimo 60.

CHECAGEM DE PROCEDENCIA. Alem das metricas, o script confere que a previsao pontual
produzida aqui e a mesma ja guardada em results/. Sem isso, os intervalos poderiam vir de
modelos parecidos mas nao identicos aos do paper, e a calibracao medida nao descreveria o
que o paper reporta. E a diferenca entre medir o intervalo DESTE trabalho e medir o
intervalo de um modelo com o mesmo nome.

Uso:
    PYTHONPATH=src python scripts/run_calibracao.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cv_timeseries.data import load_and_aggregate_series  # noqa: E402
from cv_timeseries.evaluate import rolling_origin_splits  # noqa: E402

warnings.filterwarnings("ignore")
for ruidoso in ("cmdstanpy", "prophet"):
    logging.getLogger(ruidoso).setLevel(logging.CRITICAL)

SERIE = ROOT / "results" / "series" / "serie_eventos_sp_sim_real_2010_2023.csv"
BENCH = ROOT / "results" / "benchmark_sim_real_sp_2010_2023_predictions.csv"
SAIDA = ROOT / "results" / "calibracao_2010_2023"

ALPHA = 0.05          # intervalo de 95%
# Mesma semente do bootstrap do paper. Necessaria por causa do Prophet: a previsao
# pontual dele e determinista, mas a BANDA vem de amostragem posterior nao semeada, e
# sem fixar isso o PICP muda cerca de meio ponto percentual a cada execucao.
SEED = 20260817
HORIZONTE = 6
MIN_TRAIN = 60

# Especificacao IDENTICA a de src/cv_timeseries/models.py. Repetida aqui de proposito: se
# alguem mudar o modelo do benchmark e nao mudar aqui, a checagem de procedencia acusa.
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (0, 1, 1, 12)


# --------------------------------------------------------------------------- #
# modelos, cada um devolvendo (media, limite inferior, limite superior)
# --------------------------------------------------------------------------- #
def sarima_intervalo(train, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = SARIMAX(
        train,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL,
        enforce_stationarity=True,
        enforce_invertibility=True,
    ).fit(disp=False, maxiter=200)
    prev = fit.get_forecast(steps=horizon)
    ic = prev.conf_int(alpha=ALPHA)
    return (
        np.asarray(prev.predicted_mean, dtype=float),
        np.asarray(ic.iloc[:, 0], dtype=float),
        np.asarray(ic.iloc[:, 1], dtype=float),
    )


def prophet_intervalo(train, horizon):
    from prophet import Prophet

    # Semeado a cada janela, nao uma vez no inicio: assim o resultado de uma janela nao
    # depende de quantas rodaram antes dela, e rodar um subconjunto da o mesmo numero.
    np.random.seed(SEED)

    # interval_width e o UNICO parametro diferente do benchmark: o default do Prophet e
    # 0,80, e aqui precisamos de 0,95 para comparar com o nominal declarado. Nao afeta a
    # previsao pontual, que e o que a checagem de procedencia confere.
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=1 - ALPHA,
    )
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    freq = pd.infer_freq(train.index) or "MS"
    fc = m.predict(m.make_future_dataframe(periods=horizon, freq=freq)).tail(horizon)
    return (
        fc["yhat"].to_numpy(dtype=float),
        fc["yhat_lower"].to_numpy(dtype=float),
        fc["yhat_upper"].to_numpy(dtype=float),
    )


MODELOS = {"sarima": sarima_intervalo, "prophet": prophet_intervalo}


# --------------------------------------------------------------------------- #
# metricas
# --------------------------------------------------------------------------- #
def interval_score(y, lo, hi, alpha=ALPHA):
    """Gneiting e Raftery (2007), eq. 43. Menor e melhor.

    Largura + penalidade de 2/alpha por unidade de distancia quando o alvo cai fora. Com
    alpha = 0,05 o fator e 40: errar por 100 obitos custa 4.000, o que e o ponto. Uma regra
    que so olhasse cobertura premiaria intervalo infinito.
    """
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return ((hi - lo)
            + (2.0 / alpha) * np.clip(lo - y, 0, None)
            + (2.0 / alpha) * np.clip(y - hi, 0, None))


# --------------------------------------------------------------------------- #
def coleta(serie):
    linhas = []
    for jid, (train, test) in enumerate(
        rolling_origin_splits(serie, horizon=HORIZONTE, min_train_size=MIN_TRAIN), start=1
    ):
        y = test.to_numpy(dtype=float)
        for nome, fn in MODELOS.items():
            try:
                mu, lo, hi = fn(train, len(test))
            except Exception as exc:
                print(f"  [AVISO] {nome} janela {jid}: {exc}", flush=True)
                continue
            for k in range(len(test)):
                linhas.append({
                    "model": nome, "window": jid, "horizon": k + 1,
                    "date": test.index[k], "y_true": y[k], "y_pred": mu[k],
                    "lo": lo[k], "hi": hi[k],
                })
        if jid % 20 == 0:
            print(f"  ... {jid}/103", flush=True)
    return pd.DataFrame(linhas)


def checa_procedencia(d):
    """A previsao pontual daqui e a mesma ja guardada no benchmark?

    Se nao for, os intervalos nao pertencem aos modelos que o paper reporta, e nenhuma
    conclusao sobre calibracao pode ser escrita a partir deles.
    """
    if not BENCH.exists():
        print(f"  [AVISO] {BENCH.name} ausente: procedencia NAO conferida")
        return None
    b = pd.read_csv(BENCH)
    b = b[b.model.isin(MODELOS)][["model", "window", "horizon", "y_pred", "y_true"]]
    j = d.merge(b, on=["model", "window", "horizon"], suffixes=("", "_bench"))
    if j.empty:
        print("  [AVISO] nenhuma linha casou com o benchmark: procedencia NAO conferida")
        return None

    out = {}
    print(f"\n  Procedencia ({len(j)} previsoes casadas com {BENCH.name}):")
    for nome, g in j.groupby("model"):
        d_pred = float(np.abs(g.y_pred - g.y_pred_bench).max())
        d_true = float(np.abs(g.y_true - g.y_true_bench).max())
        out[nome] = {"n": int(len(g)), "max_div_y_pred": d_pred, "max_div_y_true": d_true}
        veredito = "identica" if d_pred < 1e-6 else "DIVERGE"
        print(f"    {nome:9s} n={len(g):4d}  y_true div={d_true:.2e}  "
              f"y_pred div={d_pred:.2e}  -> {veredito}")
    return out


def resume(d):
    d = d.copy()
    d["dentro"] = (d.y_true >= d.lo) & (d.y_true <= d.hi)
    d["largura"] = d.hi - d.lo
    d["is"] = interval_score(d.y_true.values, d.lo.values, d.hi.values)

    print(f"\n  Calibracao a {int((1 - ALPHA) * 100)}% nominal")
    print(f"  {'modelo':<9}{'h':>4}{'PICP':>8}{'MPIW':>9}{'IS':>10}{'n':>6}")
    resumo = {}
    for nome in MODELOS:
        g = d[d.model == nome]
        if g.empty:
            continue
        por_h = {}
        for h in range(1, HORIZONTE + 1):
            gh = g[g.horizon == h]
            por_h[h] = {"picp": float(gh.dentro.mean()), "mpiw": float(gh.largura.mean()),
                        "is": float(gh["is"].mean()), "n": int(len(gh))}
            print(f"  {nome:<9}{h:>4}{por_h[h]['picp']:>8.3f}"
                  f"{por_h[h]['mpiw']:>9.0f}{por_h[h]['is']:>10.0f}{len(gh):>6}")
        resumo[nome] = {
            "picp": float(g.dentro.mean()), "mpiw": float(g.largura.mean()),
            "is": float(g["is"].mean()), "n": int(len(g)), "por_horizonte": por_h,
        }
        print(f"  {nome:<9}{'tot':>4}{resumo[nome]['picp']:>8.3f}"
              f"{resumo[nome]['mpiw']:>9.0f}{resumo[nome]['is']:>10.0f}{len(g):>6}")
    return d, resumo


def main() -> int:
    serie = load_and_aggregate_series(str(SERIE), "date", "value", "MS")
    print(f"  serie: {len(serie)} meses, {serie.index[0]:%Y-%m} a {serie.index[-1]:%Y-%m}")

    d = coleta(serie)
    proc = checa_procedencia(d)
    d, resumo = resume(d)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(f"{SAIDA}_predictions.csv", index=False)
    payload = {
        "_meta": {
            "gerado_por": "scripts/run_calibracao.py",
            "nominal": 1 - ALPHA, "janelas": int(d.window.nunique()),
            "horizonte": HORIZONTE, "min_train": MIN_TRAIN,
            "sarima_order": list(SARIMA_ORDER), "sarima_seasonal": list(SARIMA_SEASONAL),
        },
        "procedencia": proc,
        "calibracao": resumo,
    }
    Path(f"{SAIDA}_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  results/{Path(SAIDA).name}_predictions.csv")
    print(f"  results/{Path(SAIDA).name}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
