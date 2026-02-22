#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cv_timeseries.data import load_and_aggregate_series
from cv_timeseries.evaluate import mae, rmse, rolling_origin_splits, smape
from cv_timeseries.models import ProphetForecaster, SarimaForecaster, TimesFMForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark de modelos de forecasting")
    parser.add_argument("--input-csv", required=True, help="CSV com série temporal")
    parser.add_argument("--date-col", default="date", help="Nome da coluna de data")
    parser.add_argument("--value-col", default="value", help="Nome da coluna alvo")
    parser.add_argument("--freq", default="MS", help="Frequência de agregação (ex: MS, W, D)")
    parser.add_argument("--horizon", type=int, default=6, help="Horizonte de previsão")
    parser.add_argument("--min-train-size", type=int, default=36, help="Janela mínima de treino")
    parser.add_argument(
        "--models",
        default="sarima,prophet,timesfm",
        help="Lista separada por vírgula (opções: sarima, prophet, timesfm)",
    )
    parser.add_argument("--output-prefix", default="results/benchmark", help="Prefixo de saída")
    return parser.parse_args()


def build_models(model_names: list[str]):
    selected = {m.strip().lower() for m in model_names if m.strip()}
    valid = {"sarima", "prophet", "timesfm"}
    invalid = selected - valid
    if invalid:
        raise ValueError(f"Modelos inválidos: {sorted(invalid)}")

    models = []

    if "sarima" in selected:
        models.append(SarimaForecaster())

    if "prophet" in selected:
        try:
            models.append(ProphetForecaster())
        except Exception as exc:
            print(f"[WARN] Prophet indisponível: {exc}")

    if "timesfm" in selected:
        try:
            models.append(TimesFMForecaster())
        except Exception as exc:
            print(f"[WARN] TimesFM indisponível: {exc}")

    if not models:
        raise RuntimeError("Nenhum modelo disponível para rodar.")

    return models


def run_backtest(series: pd.Series, model, horizon: int, min_train_size: int):
    y_true_all = []
    y_pred_all = []
    rows = []

    for train, test in rolling_origin_splits(series, horizon=horizon, min_train_size=min_train_size):
        try:
            y_pred = model.forecast(train, horizon=len(test))
        except Exception as exc:
            print(f"[WARN] Falha em {model.name}: {exc}")
            continue

        y_true = test.to_numpy(dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_pred) != len(y_true):
            print(f"[WARN] Tamanho inválido em {model.name}: pred={len(y_pred)} true={len(y_true)}")
            continue

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        for dt, yt, yp in zip(test.index, y_true, y_pred):
            rows.append(
                {
                    "model": model.name,
                    "date": dt,
                    "y_true": yt,
                    "y_pred": yp,
                }
            )

    if not y_true_all:
        return None, pd.DataFrame(rows)

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)

    metric_row = {
        "model": model.name,
        "mae": mae(y_true_cat, y_pred_cat),
        "rmse": rmse(y_true_cat, y_pred_cat),
        "smape": smape(y_true_cat, y_pred_cat),
        "n_predictions": int(len(y_true_cat)),
    }
    return metric_row, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    series = load_and_aggregate_series(
        csv_path=args.input_csv,
        date_col=args.date_col,
        value_col=args.value_col,
        freq=args.freq,
    )

    models = build_models(args.models.split(","))
    metrics_rows = []
    preds_frames = []

    for model in models:
        print(f"[INFO] Rodando backtest para: {model.name}")
        metric_row, pred_df = run_backtest(
            series=series,
            model=model,
            horizon=args.horizon,
            min_train_size=args.min_train_size,
        )
        if metric_row is not None:
            metrics_rows.append(metric_row)
        if not pred_df.empty:
            preds_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by="smape", ascending=True)
    else:
        metrics_df = pd.DataFrame(
            columns=["model", "mae", "rmse", "smape", "n_predictions"]
        )
    preds_df = pd.concat(preds_frames, ignore_index=True) if preds_frames else pd.DataFrame()

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = output_prefix.with_name(output_prefix.name + "_metrics.csv")
    preds_path = output_prefix.with_name(output_prefix.name + "_predictions.csv")

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print(f"[INFO] Métricas salvas em: {metrics_path}")
    print(f"[INFO] Previsões salvas em: {preds_path}")


if __name__ == "__main__":
    main()
