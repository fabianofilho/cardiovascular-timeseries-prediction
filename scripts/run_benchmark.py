#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cv_timeseries.data import load_and_aggregate_series
from cv_timeseries.evaluate import mae, rmse, rolling_origin_splits, smape
from cv_timeseries.exog import build_exog_frames
from cv_timeseries.models import (
    CatBoostForecaster,
    ProphetForecaster,
    SarimaForecaster,
    TimesFMForecaster,
    XGBoostForecaster,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark de modelos de forecasting")
    parser.add_argument("--input-csv", required=True, help="CSV com série temporal")
    parser.add_argument("--date-col", default="date", help="Nome da coluna de data")
    parser.add_argument("--value-col", default="value", help="Nome da coluna alvo")
    parser.add_argument("--freq", default="MS", help="Frequência de agregação (ex: MS, W, D)")
    parser.add_argument("--horizon", type=int, default=6, help="Horizonte de previsão")
    parser.add_argument(
        "--min-train-size",
        type=int,
        default=60,
        help="Janela mínima de treino (60 = 5 anos, >=4 ciclos sazonais efetivos)",
    )
    parser.add_argument(
        "--max-train-size",
        type=int,
        default=0,
        help="0 = janela expanding; k>0 = janela deslizante com os últimos k meses",
    )
    parser.add_argument(
        "--models",
        default="sarima,prophet,timesfm",
        help="Lista separada por vírgula (opções: sarima, prophet, timesfm, xgboost, catboost)",
    )
    parser.add_argument("--output-prefix", default="results/benchmark", help="Prefixo de saída")
    parser.add_argument(
        "--exog-csv",
        default="",
        help="CSV de exógenas (coluna de data + colunas numéricas) cobrindo toda a série",
    )
    parser.add_argument(
        "--exog-cols",
        default="tmin",
        help="Colunas do exog-csv a usar, separadas por vírgula",
    )
    parser.add_argument(
        "--exog-policy",
        default="climatology",
        choices=["climatology", "lag12", "observed"],
        help=(
            "Como preencher a exógena no período previsto: climatology (média do mês-do-ano "
            "calculada só até o fim do treino, sem vazamento), lag12 (valor observado 12 meses "
            "antes, sem vazamento), observed (valor futuro real, VAZAMENTO DELIBERADO, apenas "
            "cenário-teto rotulado)"
        ),
    )
    return parser.parse_args()


def load_exog(csv_path: str, date_col: str, cols: list[str], freq: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Coluna de data '{date_col}' ausente em {csv_path}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas de exógena ausentes em {csv_path}: {missing}")
    df[date_col] = pd.to_datetime(df[date_col])
    exog = df.set_index(date_col).sort_index()[cols].astype(float)
    exog = exog.resample(freq).mean()
    if exog.isna().any().any():
        raise ValueError("Exógena contém meses faltantes após alinhamento de frequência.")
    return exog


def build_models(model_names: list[str]):
    selected = {m.strip().lower() for m in model_names if m.strip()}
    valid = {"sarima", "prophet", "timesfm", "xgboost", "catboost"}
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

    if "xgboost" in selected:
        try:
            models.append(XGBoostForecaster())
        except Exception as exc:
            print(f"[WARN] XGBoost indisponível: {exc}")

    if "catboost" in selected:
        try:
            models.append(CatBoostForecaster())
        except Exception as exc:
            print(f"[WARN] CatBoost indisponível: {exc}")

    if not models:
        raise RuntimeError("Nenhum modelo disponível para rodar.")

    return models


def run_backtest(
    series: pd.Series,
    model,
    horizon: int,
    min_train_size: int,
    model_label: str | None = None,
    exog: pd.DataFrame | None = None,
    exog_policy: str = "climatology",
    max_train_size: int | None = None,
):
    label = model_label or model.name
    y_true_all = []
    y_pred_all = []
    rows = []

    for window_id, (train, test) in enumerate(
        rolling_origin_splits(
            series,
            horizon=horizon,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        ),
        start=1,
    ):
        try:
            if exog is not None:
                exog_train, exog_future = build_exog_frames(
                    exog, train.index, test.index, exog_policy
                )
                y_pred = model.forecast(
                    train,
                    horizon=len(test),
                    exog_train=exog_train,
                    exog_future=exog_future,
                )
            else:
                y_pred = model.forecast(train, horizon=len(test))
        except Exception as exc:
            print(f"[WARN] Falha em {label} (janela {window_id}): {exc}")
            continue

        y_true = test.to_numpy(dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(y_pred) != len(y_true):
            print(f"[WARN] Tamanho inválido em {label}: pred={len(y_pred)} true={len(y_true)}")
            continue

        if not np.all(np.isfinite(y_pred)):
            print(f"[WARN] Previsão não-finita em {label} (janela {window_id}); janela descartada")
            continue

        # Diagnóstico (substitui o antigo clamp, que corrigia valores só para
        # parte dos modelos): alerta sem alterar a previsão.
        lo, hi = train.min() * 0.1, train.max() * 3.0
        if np.any((y_pred < lo) | (y_pred > hi)):
            print(
                f"[WARN] Previsão fora de [{lo:.1f}, {hi:.1f}] em {label} "
                f"(janela {window_id}); valor mantido sem clamp"
            )

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

        train_end = train.index[-1]
        for h, (dt, yt, yp) in enumerate(zip(test.index, y_true, y_pred), start=1):
            rows.append(
                {
                    "model": label,
                    "date": dt,
                    "y_true": yt,
                    "y_pred": yp,
                    "window": window_id,
                    "horizon": h,
                    "train_end": train_end,
                }
            )

    if not y_true_all:
        return None, pd.DataFrame(rows)

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)

    metric_row = {
        "model": label,
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

    exog = None
    if args.exog_csv:
        exog_cols = [c.strip() for c in args.exog_cols.split(",") if c.strip()]
        exog = load_exog(args.exog_csv, args.date_col, exog_cols, args.freq)
        missing_dates = series.index.difference(exog.index)
        if len(missing_dates) > 0:
            raise ValueError(
                f"Exógena não cobre toda a série; faltam {len(missing_dates)} meses "
                f"(ex: {missing_dates[0]})"
            )
        unsupported = [m.name for m in models if not m.supports_exog]
        if unsupported:
            print(f"[WARN] Modelos sem suporte a exógena serão pulados: {unsupported}")
            models = [m for m in models if m.supports_exog]
        if not models:
            raise RuntimeError("Nenhum modelo com suporte a exógena para rodar.")
        print(f"[INFO] Exógena ativa: cols={exog_cols} policy={args.exog_policy}")

    metrics_rows = []
    preds_frames = []

    max_train = args.max_train_size if args.max_train_size > 0 else None
    if max_train is not None:
        print(f"[INFO] Janela deslizante: treino limitado aos últimos {max_train} meses")

    for model in models:
        label = model.name
        if exog is not None:
            label = f"{label}_temp"
        if max_train is not None:
            label = f"{label}_slide{max_train}"
        print(f"[INFO] Rodando backtest para: {label}")
        metric_row, pred_df = run_backtest(
            series=series,
            model=model,
            horizon=args.horizon,
            min_train_size=args.min_train_size,
            model_label=label,
            exog=exog,
            exog_policy=args.exog_policy,
            max_train_size=max_train,
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
