# -*- coding: utf-8 -*-
"""Tabela 5 com a especificacao melhorada: teste falsificavel da explicacao do ponto 21.

Hipotese: na Tabela 5 original a temperatura ajuda os boosters (0,258 e 0,382 pp) muito
mais que o SARIMA (0,136) e que o Prophet (-0,035) porque a covariavel e quase uma funcao
deterministica do mes, e os boosters eram os unicos sem sazonalidade explicita nas features.

Predicao: dando aos boosters um termo de calendario de Fourier (variante diffcal), o ganho
da temperatura deve encolher bastante. Se NAO encolher, a explicacao esta errada e a
temperatura carrega informacao alem da climatologia.

Politica climatology, sem vazamento, identica a do repositorio.
"""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))   # ADAPTADO: variants.py

from cv_timeseries.data import load_and_aggregate_series  # noqa: E402
from cv_timeseries.evaluate import rolling_origin_splits, smape  # noqa: E402
# ADAPTADO: build_exog_frames saiu do run_benchmark para cv_timeseries.exog.
from cv_timeseries.exog import build_exog_frames  # noqa: E402
from run_benchmark import load_exog  # noqa: E402

from variants import build_regressor, calendar_exog  # noqa: E402

warnings.filterwarnings("ignore")

SERIES = REPO / "results/series/serie_eventos_sp_sim_real_2010_2023.csv"
TEMP = REPO / "results/series/temperatura_sp_mensal_2010_2023.csv"


def forecast(train, horizon, kind, use_cal, use_diff, temp_tr=None, temp_fu=None):
    """diffcal opcionalmente com temperatura somada as exogenas."""
    from skforecast.recursive import ForecasterRecursive

    y = train.copy()
    if y.index.freq is None:
        y = y.asfreq(pd.infer_freq(y.index) or "MS")
    fut = pd.date_range(y.index[-1] + y.index.freq, periods=horizon, freq=y.index.freq)

    target = (y - y.shift(12)).dropna() if use_diff else y

    ex_tr = calendar_exog(target.index) if use_cal else None
    ex_fu = calendar_exog(fut) if use_cal else None

    if temp_tr is not None:
        t_tr = temp_tr.reindex(target.index)
        ex_tr = t_tr if ex_tr is None else pd.concat([ex_tr, t_tr], axis=1)
        ex_fu = temp_fu if ex_fu is None else pd.concat([ex_fu, temp_fu], axis=1)

    f = ForecasterRecursive(build_regressor(kind),
                            lags=max(1, min(12, len(target) - 1)))
    f.fit(y=target, exog=ex_tr)
    pred = np.asarray(f.predict(steps=horizon, exog=ex_fu), dtype=float)

    if use_diff:
        pred = pred + np.asarray(
            [y.loc[d - pd.DateOffset(months=12)] for d in fut], dtype=float)
    return pred


def main():
    s = load_and_aggregate_series(str(SERIES), "date", "value", "MS")
    exog = load_exog(str(TEMP), "date", ["tmin"], "MS")

    combos = [
        ("base",    False, False, False),
        ("base+T",  False, False, True),
        ("diffcal", True,  True,  False),
        ("diffcal+T", True, True,  True),
    ]
    acc = {(k, n): ([], []) for k in ("catboost", "xgboost") for n, _, _, _ in combos}

    for wid, (tr, te) in enumerate(
            rolling_origin_splits(s, horizon=6, min_train_size=60), 1):
        ex_tr, ex_fu = build_exog_frames(exog, tr.index, te.index, "climatology")
        for kind in ("catboost", "xgboost"):
            for name, use_cal, use_diff, use_t in combos:
                p = forecast(tr, len(te), kind, use_cal, use_diff,
                             ex_tr if use_t else None, ex_fu if use_t else None)
                acc[(kind, name)][0].append(te.to_numpy(float))
                acc[(kind, name)][1].append(p)
        if wid % 25 == 0:
            print(f"  ... {wid}/103", flush=True)

    print("\n=== Ganho da temperatura, por especificacao ===")
    print(f"{'modelo':<10}{'espec.':<10}{'sem T':>8}{'com T':>8}{'ganho':>9}")
    rows = []
    for kind in ("catboost", "xgboost"):
        for base, comt in (("base", "base+T"), ("diffcal", "diffcal+T")):
            a = smape(np.concatenate(acc[(kind, base)][0]),
                      np.concatenate(acc[(kind, base)][1]))
            b = smape(np.concatenate(acc[(kind, comt)][0]),
                      np.concatenate(acc[(kind, comt)][1]))
            print(f"{kind:<10}{base:<10}{a:8.3f}{b:8.3f}{a-b:+9.3f}")
            rows.append(dict(modelo=kind, espec=base, sem_T=a, com_T=b, ganho=a - b))
    destino = REPO / "results" / "revisao" / "temp_melhorado.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destino, index=False)   # ADAPTADO
    print(f"\n[INFO] {destino}")
    print("Predicao: o ganho em 'diffcal' deve ser bem menor que em 'base'.")


if __name__ == "__main__":
    main()
