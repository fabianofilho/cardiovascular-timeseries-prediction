#!/usr/bin/env python3
"""Busca de hiperparametros com Optuna para os boosters.

Dois modos, deliberadamente:

dev     : SEM vazamento. Otimiza num backtest interno restrito aos primeiros 60
          meses da serie, que sao treino de TODAS as 103 janelas do benchmark.
          Nenhuma data de teste e tocada. E o que um praticante honesto faria.

oracle  : COM vazamento, rotulado. Otimiza direto no sMAPE das 103 janelas de
          teste. Nao e um resultado reportavel, e o TETO do que qualquer busca
          de hiperparametros poderia entregar nesta serie. Mesmo papel da coluna
          "ceiling" da Tabela 5 do paper.

O espaco de busca inclui o numero de lags, que no paper esta fixado em 12 sem
justificativa e faz parte da pergunta em revisao.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# ADAPTADO ao versionar: no Drive estes scripts ficavam ao lado de uma copia do
# repositorio chamada "repo/". Aqui eles moram dentro do proprio repositorio.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cv_timeseries.evaluate import rolling_origin_splits, smape  # noqa: E402
from cv_timeseries.data import load_and_aggregate_series  # noqa: E402

# ADAPTADO: import movido para dentro de forecast(). variants.py ainda nao foi
# versionado, e no topo do modulo ele derrubaria tambem o modo "base", que nao usa
# calendar_exog. Ver o README desta pasta.

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SERIES = REPO / "results/series/serie_eventos_sp_sim_real_2010_2023.csv"
DEV_MONTHS = 60  # treino comum a todas as janelas do benchmark


def suggest(trial, kind):
    if kind == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            random_state=42,
            n_jobs=-1,
        )
    return dict(
        iterations=trial.suggest_int("iterations", 100, 1500, step=50),
        depth=trial.suggest_int("depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def make_regressor(kind, params):
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**params)
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params)


def forecast(train, horizon, kind, params, lags, variant):
    """Mesma mecanica de variants.forecast_variant, com params vindos do trial."""
    from skforecast.recursive import ForecasterRecursive
    if variant == "diffcal":
        from variants import calendar_exog

    y = train.copy()
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index)
    if y.index.freq is None:
        y = y.asfreq(pd.infer_freq(y.index) or "MS")
    fut = pd.date_range(y.index[-1] + y.index.freq, periods=horizon, freq=y.index.freq)

    use_diff = variant in ("diffcal",)
    use_cal = variant in ("diffcal",)

    target = (y - y.shift(12)).dropna() if use_diff else y
    usable = max(1, min(lags, len(target) - 1))

    f = ForecasterRecursive(make_regressor(kind, params), lags=usable)
    f.fit(
        y=target,
        exog=calendar_exog(target.index) if use_cal else None,
    )
    pred = np.asarray(
        f.predict(steps=horizon, exog=calendar_exog(fut) if use_cal else None),
        dtype=float,
    )
    if use_diff:
        pred = pred + np.asarray(
            [y.loc[d - pd.DateOffset(months=12)] for d in fut], dtype=float
        )
    return pred


def evaluate(series, kind, params, lags, variant, horizon, min_train, max_train=None):
    yt, yp = [], []
    for train, test in rolling_origin_splits(
        series, horizon=horizon, min_train_size=min_train, max_train_size=max_train
    ):
        try:
            p = forecast(train, len(test), kind, params, lags, variant)
        except Exception:
            return float("inf")
        if not np.all(np.isfinite(p)):
            return float("inf")
        yt.append(test.to_numpy(dtype=float))
        yp.append(p)
    if not yt:
        return float("inf")
    return smape(np.concatenate(yt), np.concatenate(yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["xgboost", "catboost"])
    ap.add_argument("--mode", required=True, choices=["dev", "oracle"])
    ap.add_argument("--variant", default="base", choices=["base", "diffcal"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="out/optuna")
    args = ap.parse_args()

    series = load_and_aggregate_series(str(SERIES), "date", "value", "MS")

    if args.mode == "dev":
        # Somente os 60 primeiros meses: treino comum a todas as janelas.
        sub = series.iloc[:DEV_MONTHS]
        space = dict(series=sub, horizon=6, min_train=36)
    else:
        space = dict(series=series, horizon=6, min_train=60)

    def objective(trial):
        params = suggest(trial, args.kind)
        lags = trial.suggest_int("lags", 6, 24)
        return evaluate(
            space["series"],
            args.kind,
            params,
            lags,
            args.variant,
            space["horizon"],
            space["min_train"],
        )

    t0 = time.time()
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260817)
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    dt = time.time() - t0

    best = dict(study.best_params)
    lags = best.pop("lags")
    params = suggest(optuna.trial.FixedTrial({**best, "lags": lags}), args.kind)

    # Avaliacao final SEMPRE no benchmark completo (103 janelas), com os
    # hiperparametros escolhidos. No modo dev isso e honesto; no oracle e o teto.
    final = evaluate(series, args.kind, params, lags, args.variant, 6, 60)

    print(
        f"[{args.mode}/{args.variant}] {args.kind}: objetivo={study.best_value:.4f}  "
        f"sMAPE no benchmark completo={final:.4f}  lags={lags}  "
        f"({args.trials} trials, {dt:.0f}s)"
    )
    print(f"    melhores params: {best}")

    out = Path(f"{args.out}_{args.kind}_{args.mode}_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(
        {
            "kind": args.kind,
            "mode": args.mode,
            "variant": args.variant,
            "objective": study.best_value,
            "smape_benchmark": final,
            "lags": lags,
            "trials": args.trials,
            "seconds": round(dt),
            **best,
        }
    ).to_json(out)
    print(f"[INFO] {out}")


if __name__ == "__main__":
    main()
