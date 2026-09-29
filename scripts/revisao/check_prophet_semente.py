# -*- coding: utf-8 -*-
"""A banda do Prophet e semeada ou nao? E o resultado reproduz em outra maquina?

Existiam duas afirmacoes contraditorias no repositorio. O run_calibracao.py semeia o
numpy a cada janela, com um comentario dizendo que isso e necessario por causa da banda.
A nota da Tabela 6 dizia o contrario: que a amostragem nao e semeada e que rodar de novo
move os limites em ate cem obitos. A pergunta foi levantada na revisao da Isabella Saade.

Este script mede as duas coisas, com o Prophet do lock (1.4.0):

  1. SEMENTE. Em tres janelas (a primeira, a do meio, a ultima), dois ajustes com a
     semente do run_calibracao antes de cada um, e dois ajustes sem semente. Se a semente
     controla a banda, o primeiro par da diferenca zero e o segundo nao.

  2. ENTRE MAQUINAS. As 103 janelas com a semente, comparadas com as previsoes guardadas
     em results/calibracao_2010_2023_predictions.csv, que vieram de outra maquina. Mede
     se o ponto e a banda reproduzem e quanto isso move PICP e sMAPE.

Saida: results/revisao/prophet_semente.json, lida por scripts/build_paper_assets.py
para a nota da Tabela 6.

Uso:
    PYTHONPATH=src python scripts/revisao/check_prophet_semente.py
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
warnings.filterwarnings("ignore")

import run_calibracao as rc  # noqa: E402

for ruidoso in ("cmdstanpy", "prophet"):
    logging.getLogger(ruidoso).disabled = True

GUARDADO = RAIZ / "results" / "calibracao_2010_2023_predictions.csv"
SAIDA = RAIZ / "results" / "revisao" / "prophet_semente.json"
JANELAS_SEMENTE = (1, 52, 103)
TOL = 0.01   # obitos; abaixo disso a diferenca e ruido de ponto flutuante


def sem_semente(train, horizon):
    """O mesmo ajuste de rc.prophet_intervalo, sem o np.random.seed."""
    from prophet import Prophet

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=1 - rc.ALPHA)
    m.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
    fc = m.predict(m.make_future_dataframe(periods=horizon, freq="MS")).tail(horizon)
    return (fc["yhat"].to_numpy(float), fc["yhat_lower"].to_numpy(float),
            fc["yhat_upper"].to_numpy(float))


def maxdif(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def smape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p))) * 100)


def main() -> int:
    import prophet

    serie = rc.load_and_aggregate_series(str(rc.SERIE), "date", "value", "MS")
    splits = list(rc.rolling_origin_splits(serie, horizon=rc.HORIZONTE,
                                           min_train_size=rc.MIN_TRAIN))

    # ---------- 1. a semente controla a banda? ----------------------------------
    semente = []
    for j in JANELAS_SEMENTE:
        tr, _ = splits[j - 1]
        a, b = rc.prophet_intervalo(tr, rc.HORIZONTE), rc.prophet_intervalo(tr, rc.HORIZONTE)
        c, d = sem_semente(tr, rc.HORIZONTE), sem_semente(tr, rc.HORIZONTE)
        linha = {
            "janela": j,
            "semeado_ponto": maxdif(a[0], b[0]),
            "semeado_banda": max(maxdif(a[1], b[1]), maxdif(a[2], b[2])),
            "sem_semente_ponto": maxdif(c[0], d[0]),
            "sem_semente_banda": max(maxdif(c[1], d[1]), maxdif(c[2], d[2])),
        }
        semente.append(linha)
        print(f"  janela {j:>3}: semeado ponto {linha['semeado_ponto']:.2e} banda "
              f"{linha['semeado_banda']:.2e} | sem semente ponto "
              f"{linha['sem_semente_ponto']:.2e} banda {linha['sem_semente_banda']:.1f}")

    # ---------- 2. reproduz em outra maquina? -----------------------------------
    linhas = []
    for j, (tr, te) in enumerate(splits, start=1):
        mu, lo, hi = rc.prophet_intervalo(tr, len(te))
        for k in range(len(te)):
            linhas.append({"window": j, "horizon": k + 1, "y": float(te.iloc[k]),
                           "mu": mu[k], "lo": lo[k], "hi": hi[k]})
    d = pd.DataFrame(linhas)
    g = pd.read_csv(GUARDADO)
    g = g[g.model == "prophet"][["window", "horizon", "y_pred", "lo", "hi"]]
    m = d.merge(g, on=["window", "horizon"], suffixes=("", "_g"))
    dif = np.maximum.reduce([(m.mu - m.y_pred).abs(), (m.lo - m.lo_g).abs(),
                             (m.hi - m.hi_g).abs()])
    picp = float(((m.y >= m.lo) & (m.y <= m.hi)).mean())
    picp_g = float(((m.y >= m.lo_g) & (m.y <= m.hi_g)).mean())
    entre = {
        "previsoes": int(len(m)),
        "janelas": int(m.window.nunique()),
        "janelas_divergentes": int(m.loc[dif > TOL, "window"].nunique()),
        "dif_max_ponto": float((m.mu - m.y_pred).abs().max()),
        "dif_max_banda": float(max((m.lo - m.lo_g).abs().max(), (m.hi - m.hi_g).abs().max())),
        "picp_aqui": picp, "picp_guardado": picp_g,
        "smape_aqui": smape(m.y, m.mu), "smape_guardado": smape(m.y, m.y_pred),
    }
    print(f"  entre maquinas: {entre['janelas_divergentes']} de {entre['janelas']} janelas "
          f"divergem, ponto ate {entre['dif_max_ponto']:.1f} obitos; PICP "
          f"{picp:.4f} contra {picp_g:.4f}; sMAPE {entre['smape_aqui']:.4f} contra "
          f"{entre['smape_guardado']:.4f}")

    payload = {
        "_meta": {
            "gerado_por": "scripts/revisao/check_prophet_semente.py",
            "semente": rc.SEED, "tolerancia_obitos": TOL,
            "ambiente": {"plataforma": platform.platform(), "python": platform.python_version(),
                         "prophet": prophet.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__},
        },
        "semente": semente,
        "semente_resumo": {
            "semeado_dif_max": max(max(s["semeado_ponto"], s["semeado_banda"]) for s in semente),
            "sem_semente_banda_min": min(s["sem_semente_banda"] for s in semente),
            "sem_semente_banda_max": max(s["sem_semente_banda"] for s in semente),
        },
        "entre_maquinas": entre,
    }
    SAIDA.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
