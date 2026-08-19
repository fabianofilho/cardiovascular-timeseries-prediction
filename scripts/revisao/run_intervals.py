#!/usr/bin/env python3
"""Intervalos de previsao nativos e calibracao empirica por horizonte.

O manuscrito afirmava que nenhum dos modelos produz intervalo de previsao, e chama a
ausencia de calibracao de "the most consequential gap in this work". SARIMA e Prophet
produzem intervalo nativamente. Este script liga os dois e mede:

  PICP  cobertura empirica -- fracao de observacoes dentro do intervalo (alvo: 0,95)
  MPIW  largura media do intervalo, em obitos/mes
  IS    interval score de Gneiting-Raftery (menor e melhor); penaliza largura e
        adiciona penalidade proporcional a distancia quando o alvo cai fora

Mesmo protocolo: 103 janelas, horizonte 6, treino minimo 60, janela expansiva.
"""
import sys, warnings, logging
from pathlib import Path
import numpy as np, pandas as pd

# ADAPTADO ao versionar: no Drive estes scripts ficavam ao lado de uma copia do
# repositorio chamada "repo/". Aqui eles moram dentro do proprio repositorio.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cv_timeseries.data import load_and_aggregate_series
from cv_timeseries.evaluate import rolling_origin_splits

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

ALPHA = 0.05  # intervalo de 95%
SERIES = REPO / "results/series/serie_eventos_sp_sim_real_2010_2023.csv"


def sarima_pi(train, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12),
                  enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=200)
    f = fit.get_forecast(steps=h)
    ci = f.conf_int(alpha=ALPHA)
    return np.asarray(f.predicted_mean, float), np.asarray(ci.iloc[:, 0], float), np.asarray(ci.iloc[:, 1], float)


def prophet_pi(train, h):
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=1 - ALPHA)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = m.predict(m.make_future_dataframe(periods=h, freq="MS")).tail(h)
    return (fc["yhat"].to_numpy(float), fc["yhat_lower"].to_numpy(float),
            fc["yhat_upper"].to_numpy(float))


def interval_score(y, lo, hi, alpha=ALPHA):
    return ((hi - lo)
            + (2.0 / alpha) * (lo - y) * (y < lo)
            + (2.0 / alpha) * (y - hi) * (y > hi))


def main():
    s = load_and_aggregate_series(str(SERIES), "date", "value", "MS")
    rows = []
    for wid, (tr, te) in enumerate(rolling_origin_splits(s, horizon=6, min_train_size=60), 1):
        y = te.to_numpy(float)
        for name, fn in (("sarima", sarima_pi), ("prophet", prophet_pi)):
            try:
                mu, lo, hi = fn(tr, len(te))
            except Exception as e:
                print(f"[WARN] {name} janela {wid}: {e}", flush=True); continue
            for k in range(len(te)):
                rows.append(dict(model=name, window=wid, horizon=k + 1, date=te.index[k],
                                 y_true=y[k], y_pred=mu[k], lo=lo[k], hi=hi[k]))
        if wid % 25 == 0:
            print(f"  ... {wid}/103", flush=True)

    d = pd.DataFrame(rows)
    d["dentro"] = (d.y_true >= d.lo) & (d.y_true <= d.hi)
    d["largura"] = d.hi - d.lo
    d["is"] = interval_score(d.y_true.values, d.lo.values, d.hi.values)
    d.to_csv("out/intervals_predictions.csv", index=False)

    print(f"\n=== Calibracao a {int((1-ALPHA)*100)}% nominal ===")
    print(f"{'modelo':<10}{'h':>3}{'PICP':>9}{'MPIW':>9}{'IS':>10}")
    for m in ["sarima", "prophet"]:
        g = d[d.model == m]
        for h in range(1, 7):
            gh = g[g.horizon == h]
            print(f"{m:<10}{h:>3}{gh.dentro.mean():9.3f}{gh.largura.mean():9.0f}{gh['is'].mean():10.0f}")
        print(f"{m:<10}{'all':>3}{g.dentro.mean():9.3f}{g.largura.mean():9.0f}{g['is'].mean():10.0f}")
    print("\n[INFO] out/intervals_predictions.csv")


if __name__ == "__main__":
    main()
