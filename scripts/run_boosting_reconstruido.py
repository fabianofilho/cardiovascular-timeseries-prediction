"""XGBoost e CatBoost na série reconstruída, comparados nos MESMOS pontos de teste.

Contexto e limites, que valem mais que o resultado:

* A série reconstruída tem 36 meses (2021-01 a 2023-12), não os 60 da extração original,
  porque os 24 primeiros meses nunca entraram em janela de teste e não são recuperáveis do
  CSV de previsões. Ver `scripts/rebuild_series_from_predictions.py` e LAB-62.
* Com 12 defasagens, sobram 24 linhas supervisionadas. Isso permite 7 janelas de rolling
  origin com horizonte 6, contra as 31 do benchmark oficial.
* A comparação é feita só nos pontos (origem, horizonte) que existem nos dois lados, o que
  isola a diferença de protocolo. O que não dá para igualar é o histórico de treino: os três
  modelos incumbentes viram 2019 e 2020, o boosting não. Qualquer diferença medida aqui
  carrega essa desvantagem embutida, e o relatório diz isso em vez de esconder.

Hiperparâmetros são pré-especificados e conservadores, não buscados. Com poucas dezenas de
linhas, busca de hiperparâmetro lê ruído (aprendizado 6 de `aprendizados-pipeline-agentes.md`).
Duas configurações de defasagem são rodadas: a do `TODO.md` (1 a 12) e uma enxuta (1, 2, 3, 12)
proporcional ao tamanho da amostra.

Saídas:
    results/benchmark_boosting_reconstruido.csv  (N5)
    results/shap_lags_xgboost.csv                (N6)

Uso:
    python scripts/run_boosting_reconstruido.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import DMatrix, XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SERIE_CSV = ROOT / "data" / "processed" / "serie_sim_sp_2021_2023_reconstruida.csv"
INDEX_CSV = ROOT / "results" / "predictions_indexed.csv"
OUT_BENCH = ROOT / "results" / "benchmark_boosting_reconstruido.csv"
OUT_SHAP = ROOT / "results" / "shap_lags_xgboost.csv"

HORIZON = 6
MIN_TRAIN_ROWS = 12
SEED = 20260801
B = 10_000
CONFIGS = {"lags_1_12": list(range(1, 13)), "lags_1_2_3_12": [1, 2, 3, 12]}


def smape_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))


def make_model(nome: str):
    """Hiperparâmetros fixados a priori: árvores rasas, taxa baixa, sem busca."""
    if nome == "xgboost":
        return XGBRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=SEED, verbosity=0,
        )
    return CatBoostRegressor(
        iterations=300, depth=2, learning_rate=0.05,
        random_seed=SEED, verbose=False, allow_writing_files=False,
    )


def supervised(valores: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Matriz de defasagens: X[i] são os lags do alvo y[i], idx[i] é o índice do alvo na série."""
    maxlag = max(lags)
    linhas, alvos, idx = [], [], []
    for t in range(maxlag, len(valores)):
        linhas.append([valores[t - lag] for lag in lags])
        alvos.append(valores[t])
        idx.append(t)
    return np.array(linhas), np.array(alvos), np.array(idx)


def prever_recursivo(modelo, historico: list[float], lags: list[int], passos: int) -> list[float]:
    """Previsão recursiva: a previsão de h alimenta as defasagens de h+1."""
    hist = list(historico)
    saida = []
    for _ in range(passos):
        x = np.array([[hist[-lag] for lag in lags]])
        p = float(modelo.predict(x)[0])
        saida.append(p)
        hist.append(p)
    return saida


def main() -> int:
    serie = pd.read_csv(SERIE_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    valores = serie["value"].to_numpy(dtype=float)
    datas = serie["date"].to_numpy()
    inc = pd.read_csv(INDEX_CSV, parse_dates=["date"])

    # origem de cada janela dos incumbentes = mês anterior à primeira data prevista
    primeira = inc.groupby(["model", "window"])["date"].min().reset_index()

    registros, resumo = [], []
    modelo_final, X_final, lags_final = None, None, None

    for cfg, lags in CONFIGS.items():
        X, y, idx_alvo = supervised(valores, lags)
        # origens possíveis: precisam de MIN_TRAIN_ROWS linhas de treino e 6 meses à frente
        origens = [i for i in range(MIN_TRAIN_ROWS - 1, len(y))
                   if idx_alvo[i] + HORIZON < len(valores)]
        for nome in ("xgboost", "catboost"):
            for o in origens:
                modelo = make_model(nome)
                modelo.fit(X[: o + 1], y[: o + 1])
                t_origem = idx_alvo[o]
                preds = prever_recursivo(modelo, list(valores[: t_origem + 1]), lags, HORIZON)
                for h, p in enumerate(preds, start=1):
                    registros.append({
                        "config": cfg, "model": nome,
                        "origin_date": pd.Timestamp(datas[t_origem]),
                        "horizon": h, "date": pd.Timestamp(datas[t_origem + h]),
                        "y_true": valores[t_origem + h], "y_pred": p,
                    })
                if nome == "xgboost" and cfg == "lags_1_2_3_12" and o == origens[-1]:
                    modelo_final, X_final, lags_final = modelo, X[: o + 1], lags

    bo = pd.DataFrame(registros)
    bo["first_date"] = bo["origin_date"] + pd.DateOffset(months=1)

    # pontos de teste comuns: mesma janela (pela primeira data prevista) e mesmo horizonte
    janelas_boost = sorted(bo["first_date"].unique())
    inc_map = primeira[primeira["date"].isin(janelas_boost)]
    inc_sel = inc.merge(
        inc_map.rename(columns={"date": "first_date"}), on=["model", "window"], how="inner"
    )

    comum = set(zip(inc_sel["first_date"], inc_sel["horizon"]))
    bo_sel = bo[[(f, h) in comum for f, h in zip(bo["first_date"], bo["horizon"])]]

    n_janelas = len(janelas_boost)
    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, n_janelas, size=(B, n_janelas))

    def bloco(df: pd.DataFrame, chave: str) -> dict:
        """sMAPE pontual e IC95% reamostrando janelas inteiras."""
        piv = (df.assign(s=smape_vec(df["y_true"].to_numpy(), df["y_pred"].to_numpy()))
                 .pivot_table(index="first_date", columns="horizon", values="s"))
        mat = piv.to_numpy()
        b = mat[idx_boot].mean(axis=(1, 2))
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {"rotulo": chave, "smape": float(np.mean(mat)), "ci_low": lo, "ci_high": hi,
                "ci_width": hi - lo, "n_windows": mat.shape[0], "n_predictions": mat.size}

    for cfg in CONFIGS:
        for nome in ("xgboost", "catboost"):
            sub = bo_sel[(bo_sel.config == cfg) & (bo_sel.model == nome)]
            r = bloco(sub, f"{nome} ({cfg})")
            r.update({"model": nome, "config": cfg, "familia": "boosting",
                      "treino_desde": "2021-01"})
            resumo.append(r)
    for nome in ("timesfm", "sarima", "prophet"):
        sub = inc_sel[inc_sel.model == nome].rename(columns={"horizon": "horizon"})
        r = bloco(sub, f"{nome} (benchmark oficial)")
        r.update({"model": nome, "config": "original", "familia": "incumbente",
                  "treino_desde": "2019-01"})
        resumo.append(r)

    res = pd.DataFrame(resumo).sort_values("smape").reset_index(drop=True)
    res.to_csv(OUT_BENCH, index=False)

    print(f"N5  Comparação nos mesmos {n_janelas} janelas x {HORIZON} horizontes "
          f"= {n_janelas * HORIZON} pontos de teste")
    print("    (2023-01 a 2023-12; boosting treina desde 2021-01, incumbentes desde 2019-01)\n")
    print(f"{'modelo':<32}{'sMAPE':>8}{'IC95%':>22}{'largura':>10}")
    for _, r in res.iterrows():
        ic = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        print(f"{r.rotulo:<32}{r.smape:>8.2f}{ic:>22}{r.ci_width:>10.2f}")
    amp = res.smape.max() - res.smape.min()
    larg = res.ci_width.mean()
    print(f"\nAmplitude entre os {len(res)} modelos: {amp:.2f} pp")
    print(f"Largura média do IC:            {larg:.2f} pp")
    print(f"Razão amplitude/largura:        {amp / larg:.2f}"
          f"  ({'ranking informativo' if amp > larg else 'ranking dentro do ruído'})")

    # ---------- N6: TreeSHAP dos lags ----------
    booster = modelo_final.get_booster()
    dmat = DMatrix(X_final, feature_names=[f"lag{l}" for l in lags_final])
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = np.asarray(contribs)[:, :-1]  # última coluna é o valor base
    lin = LinearRegression().fit(X_final, modelo_final.predict(X_final))

    linhas = []
    for j, lag in enumerate(lags_final):
        corr = float(np.corrcoef(X_final[:, j], shap_vals[:, j])[0, 1])
        coef = float(lin.coef_[j])
        linhas.append({
            "lag": lag,
            "mean_abs_shap": float(np.abs(shap_vals[:, j]).mean()),
            "direcao_shap": "positiva" if corr > 0 else "negativa",
            "corr_lag_vs_shap": corr,
            "coef_linear": coef,
            "direcao_linear": "positiva" if coef > 0 else "negativa",
            "divergencia_de_sinal": bool(np.sign(corr) != np.sign(coef)),
        })
    shap_df = pd.DataFrame(linhas).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUT_SHAP, index=False)

    print(f"\n\nN6  TreeSHAP do XGBoost final ({len(X_final)} linhas de treino, "
          f"lags {lags_final})\n")
    print(f"{'lag':<6}{'|SHAP| médio':>14}{'direção SHAP':>15}{'direção linear':>17}{'diverge?':>10}")
    for _, r in shap_df.iterrows():
        print(f"{int(r.lag):<6}{r.mean_abs_shap:>14.1f}{r.direcao_shap:>15}"
              f"{r.direcao_linear:>17}{'SIM' if r.divergencia_de_sinal else '-':>10}")

    print(f"\nGravado: {OUT_BENCH.relative_to(ROOT)}")
    print(f"Gravado: {OUT_SHAP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
