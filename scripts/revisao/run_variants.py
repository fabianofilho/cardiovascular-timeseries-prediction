#!/usr/bin/env python3
"""Roda as variantes de feature engineering sob o protocolo do paper."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ADAPTADO ao versionar: no Drive estes scripts ficavam ao lado de uma copia do
# repositorio chamada "repo/". Aqui eles moram dentro do proprio repositorio.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cv_timeseries.data import load_and_aggregate_series  # noqa: E402
from cv_timeseries.evaluate import mae, rmse, rolling_origin_splits, smape  # noqa: E402

from variants import forecast_variant  # noqa: E402

SERIES = REPO / "results/series/serie_eventos_sp_sim_real_2010_2023.csv"


def run(series, kind, variant, horizon, min_train, max_train):
    rows = []
    yt_all, yp_all = [], []
    for wid, (train, test) in enumerate(
        rolling_origin_splits(
            series, horizon=horizon, min_train_size=min_train, max_train_size=max_train
        ),
        start=1,
    ):
        y_pred = forecast_variant(train, horizon=len(test), kind=kind, variant=variant)
        y_true = test.to_numpy(dtype=float)
        if len(y_pred) != len(y_true) or not np.all(np.isfinite(y_pred)):
            print(f"[WARN] janela {wid} descartada ({kind}/{variant})")
            continue
        yt_all.append(y_true)
        yp_all.append(y_pred)
        for h, (dt, a, b) in enumerate(zip(test.index, y_true, y_pred), start=1):
            rows.append(
                dict(
                    model=f"{kind}_{variant}",
                    date=dt,
                    y_true=a,
                    y_pred=b,
                    window=wid,
                    horizon=h,
                    train_end=train.index[-1],
                )
            )
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    metric = dict(
        model=f"{kind}_{variant}",
        mae=mae(yt, yp),
        rmse=rmse(yt, yp),
        smape=smape(yt, yp),
        n_predictions=int(len(yt)),
    )
    return metric, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="xgboost,catboost")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--min-train-size", type=int, default=60)
    ap.add_argument("--max-train-size", type=int, default=0)
    ap.add_argument("--out", default="out/variants")
    args = ap.parse_args()

    series = load_and_aggregate_series(str(SERIES), "date", "value", "MS")
    max_train = args.max_train_size or None

    metrics, preds = [], []
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
            t0 = time.time()
            m, p = run(
                series, kind, variant, args.horizon, args.min_train_size, max_train
            )
            m["seconds"] = round(time.time() - t0, 1)
            print(
                f"[OK] {m['model']:<26} sMAPE={m['smape']:.4f}  MAE={m['mae']:.1f}  "
                f"n={m['n_predictions']}  ({m['seconds']}s)"
            )
            metrics.append(m)
            preds.append(p)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(f"{out}_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(f"{out}_predictions.csv", index=False)
    print(f"[INFO] salvo em {out}_metrics.csv / {out}_predictions.csv")


if __name__ == "__main__":
    main()
