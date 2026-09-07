#!/usr/bin/env python3
"""Prophet com temperatura via add_regressor, e baselines ingenuas.

Motivo: o manuscrito afirma (secao 2.6) que apenas SARIMA, XGBoost e CatBoost
"natively accept exogenous regressors" e que Prophet foi excluido da Tabela 5.
Prophet tem add_regressor() nativo. Este script testa o que o paper deixou de
testar, com a MESMA politica climatology sem vazamento usada no repositorio.

Tambem gera naive, seasonal naive e seasonal naive com drift, ausentes do paper.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ADAPTADO ao versionar: no Drive estes scripts ficavam ao lado de uma copia do
# repositorio chamada "repo/". Aqui eles moram dentro do proprio repositorio.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cv_timeseries.data import load_and_aggregate_series  # noqa: E402
from cv_timeseries.evaluate import rolling_origin_splits  # noqa: E402
# ADAPTADO: build_exog_frames saiu do run_benchmark para cv_timeseries.exog, para poder
# ser testado sem arrastar as dependencias dos modelos. load_exog continua no script.
from cv_timeseries.exog import build_exog_frames  # noqa: E402
from run_benchmark import load_exog  # noqa: E402

warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

SERIES = REPO / "results/series/serie_eventos_sp_sim_real_2010_2023.csv"
TEMP = REPO / "results/series/temperatura_sp_mensal_2010_2023.csv"


def prophet_forecast(train, horizon, exog_train=None, exog_future=None):
    from prophet import Prophet

    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
    )
    if exog_train is not None:
        for c in exog_train.columns:
            m.add_regressor(c)
            df[c] = exog_train[c].to_numpy()
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="MS")
    if exog_train is not None:
        full = pd.concat([exog_train, exog_future])
        for c in exog_train.columns:
            future[c] = full[c].reindex(future.ds).to_numpy()
    return m.predict(future).tail(horizon)["yhat"].to_numpy(dtype=float)


def main():
    series = load_and_aggregate_series(str(SERIES), "date", "value", "MS")
    exog = load_exog(str(TEMP), "date", ["tmin"], "MS")

    rows = []
    for wid, (tr, te) in enumerate(
        rolling_origin_splits(series, horizon=6, min_train_size=60), start=1
    ):
        h = len(te)
        yt = te.to_numpy(dtype=float)

        # --- baselines ingenuas (ausentes do paper) ---
        drift = (tr.iloc[-1] - tr.iloc[-13]) / 12.0
        preds = {
            "naive": np.repeat(tr.iloc[-1], h),
            "snaive": np.array([tr.iloc[-12 + k - 1] for k in range(1, h + 1)]),
            "snaive_drift": np.array(
                [tr.iloc[-12 + k - 1] + drift * k for k in range(1, h + 1)]
            ),
        }

        # --- Prophet sem e com temperatura (politica climatology) ---
        preds["prophet_repro"] = prophet_forecast(tr, h)
        ex_tr, ex_fu = build_exog_frames(exog, tr.index, te.index, "climatology")
        preds["prophet_temp"] = prophet_forecast(tr, h, ex_tr, ex_fu)
        ex_tr_o, ex_fu_o = build_exog_frames(exog, tr.index, te.index, "observed")
        preds["prophet_temp_ceiling"] = prophet_forecast(tr, h, ex_tr_o, ex_fu_o)

        for name, p in preds.items():
            for k, (dt, a, b) in enumerate(zip(te.index, yt, np.asarray(p, float)), 1):
                rows.append(
                    dict(
                        model=name,
                        date=dt,
                        y_true=a,
                        y_pred=b,
                        window=wid,
                        horizon=k,
                        train_end=tr.index[-1],
                    )
                )
        if wid % 20 == 0:
            print(f"  ... janela {wid}/103", flush=True)

    out = pd.DataFrame(rows)
    destino = REPO / "results" / "revisao" / "prophet_exog_predictions.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destino, index=False)   # ADAPTADO: saida versionada

    def sm(g):
        return (
            200 * np.abs(g.y_pred - g.y_true) / (np.abs(g.y_true) + np.abs(g.y_pred))
        ).mean()

    print("\n=== resultado ===")
    for m in out.model.unique():
        g = out[out.model == m]
        print(f"  {m:<24} sMAPE={sm(g):6.3f}  MAE={np.abs(g.y_pred-g.y_true).mean():7.1f}")
    print(f"\n[INFO] {destino}")   # ADAPTADO: saida versionada


if __name__ == "__main__":
    main()
