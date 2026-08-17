#!/usr/bin/env python3
"""Gera tabelas, figuras e o JSON de conferencia do manuscrito.

Regra que este script existe para garantir: nenhum numero entra no manuscrito por
digitacao. Tabelas em `paper/tables/*.tex` e figuras em `paper/figures/*.pdf` sao
artefatos gerados, e todo valor citado na prosa sai de `paper/verified_numbers.json`.

Fonte primaria: os CSVs de PREVISAO (`*_predictions.csv`), com y_true e y_pred por
janela e horizonte, mais a serie e a temperatura em `results/series/`. As metricas
agregadas e os intervalos sao RECALCULADOS aqui, nao lidos dos `*_metrics.csv` nem dos
`uncertainty_*.csv`, justamente para que uma divergencia de agregacao apareca.

Uso:
    PYTHONPATH=src python scripts/build_paper_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"

B = 10_000
SEED = 20260817
AVISO = "% gerado por scripts/build_paper_assets.py -- nao editar a mao\n"

ROTULO = {
    "prophet": "Prophet", "sarima": "SARIMA", "timesfm": "TimesFM",
    "xgboost": "XGBoost", "catboost": "CatBoost",
}
COR = {
    "prophet": "#4C72B0", "sarima": "#DD8452", "timesfm": "#55A868",
    "xgboost": "#C44E52", "catboost": "#8172B3",
}
TOP3 = ["prophet", "sarima", "timesfm"]
ORDEM = ["prophet", "sarima", "timesfm", "catboost", "xgboost"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ metricas

def smape_vec(yt, yp):
    return 200.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp))


def matriz(df, modelo):
    """Previsoes de um modelo como matriz (janela x horizonte)."""
    g = df[df.model == modelo].sort_values(["window", "horizon"])
    nw, nh = g.window.nunique(), g.horizon.nunique()
    if len(g) != nw * nh:
        raise ValueError(f"{modelo}: {len(g)} previsoes, esperado {nw}x{nh}")
    return (g.y_true.to_numpy().reshape(nw, nh), g.y_pred.to_numpy().reshape(nw, nh))


def dm_test(d, h):
    """Diebold-Mariano com correcao de Harvey, Leybourne e Newbold."""
    n = len(d)
    db = d.mean()
    var = np.sum((d - db) ** 2) / n
    for lag in range(1, h):
        var += 2.0 * np.sum((d[lag:] - db) * (d[:-lag] - db)) / n
    if var <= 0:
        return float("nan"), float("nan")
    dm = db / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), df=n - 1)))


def bootstrap_janelas(sm_por_modelo, n_win, seed=SEED, b=B):
    """Bootstrap reamostrando JANELAS INTEIRAS, mesmas janelas para todos os modelos.

    As previsoes nao sao independentes: vem de janelas sobrepostas, e dentro de cada
    janela os horizontes compartilham a origem. Reamostrar previsao a previsao daria
    intervalo falsamente estreito.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_win, size=(b, n_win))
    return {m: sm[idx].mean(axis=(1, 2)) for m, sm in sm_por_modelo.items()}


def ic(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ LaTeX

def num(x, casas=2):
    return f"{x:.{casas}f}"


def sgn(x, casas=3):
    """Numero com sinal, em modo matematico: fora dele o '-' vira hifen, nao menos."""
    return f"${x:+.{casas}f}$"


def intervalo(lo, hi, casas=3):
    """Intervalo inteiro em modo matematico, pelo mesmo motivo."""
    return f"$[{lo:+.{casas}f}, {hi:+.{casas}f}]$"


def escreve_tabela(nome, corpo):
    (TAB / f"{nome}.tex").write_text(AVISO + corpo, encoding="utf-8")
    print(f"  tabela  paper/tables/{nome}.tex")


def salva_fig(fig, nome):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{nome}.pdf")
    fig.savefig(FIG / f"{nome}.png", dpi=150)
    plt.close(fig)
    print(f"  figura  paper/figures/{nome}.pdf (+ .png)")


# ------------------------------------------------------------------ main

def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    V: dict = {"_meta": {
        "gerado_por": "scripts/build_paper_assets.py",
        "bootstrap": "percentil, reamostragem de janelas inteiras do rolling origin",
        "B": B, "seed": SEED,
        "dm": "Diebold-Mariano, perda = erro absoluto, correcao de Harvey-Leybourne-Newbold",
        "criterio_pre_declarado": "IC95% da diferenca exclui zero E p_DM < 0.05 em >= 3 de 6 horizontes",
    }}

    # ---------- fontes primarias ----------
    serie = pd.read_csv(RES / "series/serie_eventos_sp_sim_real_2010_2023.csv",
                        parse_dates=["date"]).set_index("date")["value"]
    temp = pd.read_csv(RES / "series/temperatura_sp_mensal_2010_2023.csv",
                       parse_dates=["date"]).set_index("date")
    meta = json.loads((RES / "series/sim_real_extraction_metadata_2010_2023.json").read_text())
    base = pd.read_csv(RES / "benchmark_sim_real_sp_2010_2023_predictions.csv", parse_dates=["date"])
    exog = pd.read_csv(RES / "benchmark_exog_temp_climatology_predictions.csv", parse_dates=["date"])
    teto = pd.read_csv(RES / "benchmark_exog_temp_observed_predictions.csv", parse_dates=["date"])
    slide = pd.read_csv(RES / "benchmark_slide60_2010_2023_predictions.csv", parse_dates=["date"])
    antigo = pd.read_csv(RES / "benchmark_sim_real_sp_2019_2023_predictions.csv", parse_dates=["date"])

    print("Gerando assets do manuscrito")

    # ---------- serie ----------
    perfil = serie.groupby(serie.index.month).mean()
    V["dados"] = {
        "registros_brutos": int(meta["rows_downloaded"]),
        "registros_cv": int(meta["rows_after_cid_filter"]),
        "n_meses": int(len(serie)),
        "inicio": serie.index.min().strftime("%Y-%m"),
        "fim": serie.index.max().strftime("%Y-%m"),
        "media_mensal": float(serie.mean()),
        "mediana_mensal": float(serie.median()),
        "min_mensal": float(serie.min()), "min_mes": serie.idxmin().strftime("%Y-%m"),
        "max_mensal": float(serie.max()), "max_mes": serie.idxmax().strftime("%Y-%m"),
        "obitos_2010": float(serie[:12].sum()), "obitos_2023": float(serie[-12:].sum()),
        "pico_mes": int(perfil.idxmax()), "pico_valor": float(perfil.max()),
        "vale_mes": int(perfil.idxmin()), "vale_valor": float(perfil.min()),
        "amplitude_sazonal_pct": float(100 * (perfil.max() - perfil.min()) / perfil.mean()),
        "temp_min_media": float(temp.tmin.mean()),
        "temp_meses": int(len(temp)),
    }

    # ---------- Tabela 1: desempenho agregado ----------
    nw = base.window.nunique()
    nh = base.horizon.nunique()
    sm, ae, se = {}, {}, {}
    for m in ORDEM:
        yt, yp = matriz(base, m)
        sm[m] = smape_vec(yt, yp)
        ae[m] = np.abs(yp - yt)
        se[m] = (yp - yt) ** 2
    boot = bootstrap_janelas(sm, nw)

    V["backtest"] = {"n_janelas": int(nw), "n_horizontes": int(nh),
                     "n_previsoes_por_modelo": int(nw * nh), "min_train": 60, "horizonte": 6}
    V["tabela1"] = {}
    for m in ORDEM:
        lo, hi = ic(boot[m])
        V["tabela1"][m] = {
            "mae": float(ae[m].mean()), "rmse": float(np.sqrt(se[m].mean())),
            "smape": float(sm[m].mean()), "ic_low": lo, "ic_high": hi, "ic_width": hi - lo,
        }
    melhor = {k: min(ORDEM, key=lambda m: V["tabela1"][m][k]) for k in ("mae", "rmse", "smape")}

    linhas = []
    for m in ORDEM:
        d = V["tabela1"][m]
        cel = []
        for k, casas in (("mae", 1), ("rmse", 1), ("smape", 2)):
            s = num(d[k], casas)
            cel.append(f"\\textbf{{{s}}}" if melhor[k] == m else s)
        cel.append(f"$[{d['ic_low']:.2f}, {d['ic_high']:.2f}]$")
        cel.append(num(d["ic_width"]))
        linhas.append(f"{ROTULO[m]} & " + " & ".join(cel) + " \\\\")
    escreve_tabela("tab1_desempenho", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\caption{{Forecasting accuracy for monthly cardiovascular mortality in the state of
S\\~ao Paulo, Brazil, 2010--2023. All models were evaluated on the same
{nw * nh} out-of-sample forecasts from {nw} rolling origin windows with a
{nh}-month horizon.}}
\\label{{tab:desempenho}}
\\begin{{tabular}}{{lrrrcr}}
\\toprule
Model & MAE & RMSE & sMAPE (\\%) & 95\\% CI of sMAPE & Width \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
Bold marks the best value in each metric column. MAE and RMSE are in deaths per month;
sMAPE is symmetric mean absolute percentage error. Intervals are percentile bootstrap
with $B={B:,}$ replicates (seed {SEED}) resampling whole rolling origin windows rather
than individual forecasts, because the six horizons within a window share a training
origin and are not independent; resampling forecast by forecast would understate the
interval. The same resampled windows were applied to every model in each replicate,
which preserves pairing for the comparisons in Table~\\ref{{tab:pares}}. The interval
describes the uncertainty of the metric, not of an individual forecast: none of these
models produces a prediction interval, so forecast calibration was not assessed.
\\end{{minipage}}
\\end{{table}}
""".replace("{,}", "{,}"))

    # ---------- Tabela 2: diferencas pareadas do top-3 ----------
    V["tabela2"] = {}
    linhas = []
    for i, a in enumerate(TOP3):
        for b_ in TOP3[i + 1:]:
            dif = sm[a].mean() - sm[b_].mean()
            bd = boot[a] - boot[b_]
            lo, hi = ic(bd)
            p = min(2 * min((bd >= 0).mean(), (bd <= 0).mean()), 1.0)
            ps = [dm_test(ae[a][:, h] - ae[b_][:, h], h + 1)[1] for h in range(nh)]
            nsig = int(np.sum(np.array(ps) < 0.05))
            chave = f"{a}_menos_{b_}"
            V["tabela2"][chave] = {"dif": float(dif), "ic_low": lo, "ic_high": hi,
                                   "p_boot": float(p), "dm_sig": nsig, "dm_p_min": float(np.nanmin(ps))}
            linhas.append(
                f"{ROTULO[a]} $-$ {ROTULO[b_]} & {sgn(dif)} & "
                f"{intervalo(lo, hi)} & {p:.3f} & {nsig} of {nh} \\\\"
            )
    escreve_tabela("tab2_pares", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Paired differences in sMAPE among the three leading models. A positive
difference means the first model has the larger error.}}
\\label{{tab:pares}}
\\begin{{tabular}}{{lrcrc}}
\\toprule
Comparison & Difference (pp) & 95\\% CI & $p$ (bootstrap) & DM cells $p<0.05$ \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
pp, percentage points; DM, Diebold-Mariano test with the Harvey-Leybourne-Newbold
small-sample correction, applied separately at each of the six forecast horizons with
absolute error as the loss function. Bootstrap differences were computed within
replicate on identical resampled windows, so the interval is for the difference itself
rather than a comparison of overlapping marginal intervals. Under the criterion fixed
before the analysis, a difference counts as established only if the interval excludes
zero and DM reaches $p<0.05$ in at least three of the six horizons; no comparison in
this table meets either condition.
\\end{{minipage}}
\\end{{table}}
""")

    # ---------- Tabela 3: mesmas datas de teste ----------
    rec = base[(base.date >= "2021-01-01") & (base.date <= "2023-12-31")]
    V["tabela3"] = {}
    linhas = []
    for m in TOP3:
        a = antigo[antigo.model == m]
        n = rec[rec.model == m]
        s_ant = float(smape_vec(a.y_true.to_numpy(), a.y_pred.to_numpy()).mean())
        s_novo = float(smape_vec(n.y_true.to_numpy(), n.y_pred.to_numpy()).mean())
        V["tabela3"][m] = {"smape_curta": s_ant, "smape_longa": s_novo,
                           "ganho": s_ant - s_novo, "n_curta": len(a), "n_longa": len(n)}
        linhas.append(f"{ROTULO[m]} & {s_ant:.2f} & {s_novo:.2f} & {s_ant - s_novo:.2f} \\\\")
    escreve_tabela("tab3_serie", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Effect of training history on the same test dates (January 2021 to December
2023). Only the amount of training data differs between columns.}}
\\label{{tab:serie}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Model & sMAPE, 24--54 months & sMAPE, 132--168 months & Gain (pp) \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
The short-history column reproduces our earlier round on 60 months of data
($n={V['tabela3']['sarima']['n_curta']}$ forecasts per model); the long-history column is
the subset of the present round falling on the same target dates
($n={V['tabela3']['sarima']['n_longa']}$). Aggregate error across the two full rounds is
not comparable, because the test period changes together with the training period; this
restriction to shared dates is the comparison that isolates training length.
\\end{{minipage}}
\\end{{table}}
""")

    # ---------- Tabela 4: expanding vs deslizante ----------
    sm_sl, ae_sl = {}, {}
    for m in ORDEM:
        yt, yp = matriz(slide, f"{m}_slide60")
        sm_sl[m] = smape_vec(yt, yp)
        ae_sl[m] = np.abs(yp - yt)
    boot_sl = bootstrap_janelas(sm_sl, nw)
    V["tabela4"] = {}
    linhas = []
    for m in ORDEM:
        e, s = float(sm[m].mean()), float(sm_sl[m].mean())
        bd = boot[m] - boot_sl[m]
        lo, hi = ic(bd)
        p = min(2 * min((bd >= 0).mean(), (bd <= 0).mean()), 1.0)
        ps = [dm_test(ae[m][:, h] - ae_sl[m][:, h], h + 1)[1] for h in range(nh)]
        nsig = int(np.sum(np.array(ps) < 0.05))
        conf = "history helps" if (hi < 0 and nsig >= 3) else ("suggestive" if hi < 0 else "indifferent")
        V["tabela4"][m] = {"expanding": e, "deslizante": s, "dif": e - s,
                           "ic_low": lo, "ic_high": hi, "p_boot": float(p),
                           "dm_sig": nsig, "veredito": conf}
        linhas.append(f"{ROTULO[m]} & {e:.2f} & {s:.2f} & {sgn(e - s, 2)} & "
                      f"{intervalo(lo, hi)} & {p:.3f} & {nsig} of {nh} \\\\")
    escreve_tabela("tab4_janela", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Expanding versus sliding 60-month training window, evaluated on identical
test origins. A negative difference means the expanding window is more accurate.}}
\\label{{tab:janela}}
\\begin{{tabular}}{{lrrrcrc}}
\\toprule
Model & Expanding & Sliding 60 & Diff. (pp) & 95\\% CI & $p$ & DM cells $p<0.05$ \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
The only difference between the two runs is how much past each model may use: the
expanding window trains from the start of the series to the origin (60 to 162 months),
the sliding window on the most recent 60 months only. Test origins, horizons and
evaluation are identical, so the bootstrap is paired by window. Under the pre-declared
criterion, only SARIMA meets both conditions.
\\end{{minipage}}
\\end{{table}}
""")

    # ---------- Tabela 5: temperatura ----------
    sm_ex, ae_ex, sm_tt = {}, {}, {}
    for m in ["sarima", "catboost", "xgboost"]:
        yt, yp = matriz(exog, f"{m}_temp")
        sm_ex[m] = smape_vec(yt, yp)
        ae_ex[m] = np.abs(yp - yt)
        yt2, yp2 = matriz(teto, f"{m}_temp")
        sm_tt[m] = smape_vec(yt2, yp2)
    boot_ex = bootstrap_janelas(sm_ex, nw)
    V["tabela5"] = {}
    linhas = []
    for m in ["sarima", "catboost", "xgboost"]:
        sem, com = float(sm[m].mean()), float(sm_ex[m].mean())
        bd = boot[m] - boot_ex[m]
        lo, hi = ic(bd)
        ps = [dm_test(ae[m][:, h] - ae_ex[m][:, h], h + 1)[1] for h in range(nh)]
        nsig = int(np.sum(np.array(ps) < 0.05))
        V["tabela5"][m] = {"sem": sem, "com": com, "ganho": sem - com,
                           "ic_low": lo, "ic_high": hi, "dm_sig": nsig,
                           "teto": float(sm_tt[m].mean()), "ganho_teto": com - float(sm_tt[m].mean())}
        linhas.append(f"{ROTULO[m]} & {sem:.2f} & {com:.2f} & {sem - com:.3f} & "
                      f"{intervalo(lo, hi)} & {nsig} of {nh} & {sm_tt[m].mean():.2f} \\\\")
    escreve_tabela("tab5_temperatura", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Effect of monthly minimum temperature as an exogenous covariate, under the
leakage-free climatology policy, and the labelled ceiling scenario.}}
\\label{{tab:temperatura}}
\\begin{{tabular}}{{lrrrcrr}}
\\toprule
Model & Without & With temp. & Gain (pp) & 95\\% CI & DM cells $p<0.05$ & Ceiling \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
Under the climatology policy the future covariate is the month-of-year mean recomputed
for each window from the exogenous series truncated at that window's training end, so no
value from after the training end is visible. The ceiling column replaces it with the
true observed future temperature, which leaks by construction and is reported only to
bound what a perfect weather forecast could add. All three gains have intervals
excluding zero, but none reaches the pre-declared threshold of DM significance in at
least three horizons, so the effect is reported as suggestive and not established.
Prophet and TimesFM do not accept exogenous regressors in this implementation and were
excluded from this comparison rather than being given an input they would ignore.
\\end{{minipage}}
\\end{{table}}
""")

    # ---------- derivados citados na prosa ----------
    pontos3 = [V["tabela1"][m]["smape"] for m in TOP3]
    larg3 = [V["tabela1"][m]["ic_width"] for m in TOP3]
    V["derivados"] = {
        "amplitude_top3": max(pontos3) - min(pontos3),
        "largura_media_top3": float(np.mean(larg3)),
        "razao_amplitude_largura": (max(pontos3) - min(pontos3)) / float(np.mean(larg3)),
        "ordem_top3": sorted(TOP3, key=lambda m: V["tabela1"][m]["smape"]),
        "gap_boosting_topo": min(V["tabela1"][m]["smape"] for m in ["xgboost", "catboost"])
                             - max(pontos3),
        "erro_relativo_melhor": 100 * min(V["tabela1"][m]["mae"] for m in ORDEM) / float(serie.mean()),
        "ganho_min_serie": min(v["ganho"] for v in V["tabela3"].values()),
        "ganho_max_serie": max(v["ganho"] for v in V["tabela3"].values()),
    }

    # DM do boosting contra o top-3
    cells, sig = 0, 0
    for b_ in ["xgboost", "catboost"]:
        for a in TOP3:
            for h in range(nh):
                _, p = dm_test(ae[b_][:, h] - ae[a][:, h], h + 1)
                cells += 1
                sig += int(p < 0.05)
    V["derivados"]["dm_boosting_vs_top3_sig"] = sig
    V["derivados"]["dm_boosting_vs_top3_total"] = cells
    # denominador alternativo, usado no README: inclui tambem o par xgboost vs catboost
    ps_xc = [dm_test(ae["xgboost"][:, h] - ae["catboost"][:, h], h + 1)[1] for h in range(nh)]
    V["derivados"]["dm_boosting_todos_pares_sig"] = sig + int(np.sum(np.array(ps_xc) < 0.05))
    V["derivados"]["dm_boosting_todos_pares_total"] = cells + nh
    dm3 = [dm_test(ae[a][:, h] - ae[b_][:, h], h + 1)[1]
           for i, a in enumerate(TOP3) for b_ in TOP3[i + 1:] for h in range(nh)]
    V["derivados"]["dm_top3_sig"] = int(np.sum(np.array(dm3) < 0.05))
    V["derivados"]["dm_top3_total"] = len(dm3)

    # Rodada anterior, para a comparacao de largura de IC. O CSV original de 2019-2023
    # nao traz window/horizon (o indice so passou a ser gravado depois); o equivalente
    # indexado esta em predictions_indexed.csv, reconstruido a partir dele.
    ant_idx = pd.read_csv(RES / "predictions_indexed.csv", parse_dates=["date"])
    nw_a = ant_idx.window.nunique()
    sm_a = {m: smape_vec(*matriz(ant_idx, m)) for m in TOP3}
    boot_a = bootstrap_janelas(sm_a, nw_a, seed=20260801)
    larg_a = [ic(boot_a[m])[1] - ic(boot_a[m])[0] for m in TOP3]
    p_a = [float(sm_a[m].mean()) for m in TOP3]
    V["rodada_anterior"] = {
        "n_janelas": int(nw_a), "n_previsoes": int(nw_a * nh),
        "amplitude": max(p_a) - min(p_a), "largura_media": float(np.mean(larg_a)),
        "ordem": sorted(TOP3, key=lambda m: float(sm_a[m].mean())),
        "fator_estreitamento": float(np.mean(larg_a)) / float(np.mean(larg3)),
        "smape": {m: float(sm_a[m].mean()) for m in TOP3},
    }

    # ---------- Figura 1: serie ----------
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(serie.index, serie.values, lw=1.0, color="#333333")
    ax.set_ylabel("Deaths per month")
    ax.set_xlabel("")
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"),
               color="#4C72B0", alpha=0.10, lw=0)
    # Rotulo acima da serie, dentro da folga criada no ylim: encostado na linha ele
    # sai cortado pela margem do PDF.
    ax.set_ylim(serie.min() * 0.93, serie.max() * 1.16)
    ax.annotate("test window shared with the earlier round",
                xy=(pd.Timestamp("2022-06-15"), serie.max() * 1.09),
                fontsize=7.5, color="#3A5A8C", ha="center", va="center")
    salva_fig(fig, "fig1_serie")

    # ---------- Figura 2: sMAPE com IC (o achado principal) ----------
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(ORDEM))[::-1]
    for i, m in enumerate(ORDEM):
        d = V["tabela1"][m]
        ax.plot([d["ic_low"], d["ic_high"]], [y[i], y[i]], lw=2.4, color=COR[m], solid_capstyle="round")
        ax.plot(d["smape"], y[i], "o", ms=6, color=COR[m], zorder=3)
    ax.axvspan(min(pontos3), max(pontos3), color="#888888", alpha=0.16, lw=0, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([ROTULO[m] for m in ORDEM])
    ax.set_xlabel("sMAPE (\\%)" if False else "sMAPE (%)")
    amp = V["derivados"]["amplitude_top3"]
    lm = V["derivados"]["largura_media_top3"]
    ax.set_title(f"Spread among the leading three: {amp:.2f} pp; mean interval width: {lm:.2f} pp",
                 loc="left", fontsize=8.5)
    salva_fig(fig, "fig2_smape_ic")

    # ---------- Figura 3: erro por horizonte ----------
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for m in ORDEM:
        ax.plot(range(1, nh + 1), sm[m].mean(axis=0), "o-", ms=4, lw=1.4,
                color=COR[m], label=ROTULO[m])
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("sMAPE (%)")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.16)
    salva_fig(fig, "fig3_horizonte")

    # ---------- Figura 4: expanding vs deslizante ----------
    # Halteres, nao barras: as diferencas sao de 0,05 a 0,69 pp sobre um nivel de 4,7 a
    # 6,9, entao barras a partir do zero deixariam invisivel justamente o efeito que a
    # figura existe para mostrar. Truncar o eixo de barras seria enganoso; o haltere
    # mostra a mudanca sem truncar nada.
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(ORDEM))[::-1]
    for i, m in enumerate(ORDEM):
        e, s = V["tabela4"][m]["expanding"], V["tabela4"][m]["deslizante"]
        piora = s > e
        ax.plot([e, s], [y[i], y[i]], lw=1.6, color="#BBBBBB", zorder=1)
        ax.plot(e, y[i], "o", ms=7, color="#4C72B0", zorder=3,
                label="Expanding window" if i == 0 else None)
        ax.plot(s, y[i], "o", ms=7, color="#DD8452", zorder=3,
                label="Sliding 60 months" if i == 0 else None)
        ax.annotate(f"{s - e:+.2f} pp", xy=(max(e, s) + 0.12, y[i]), va="center",
                    fontsize=7.5, color="#C44E52" if piora else "#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels([ROTULO[m] for m in ORDEM])
    ax.set_xlabel("sMAPE (%)")
    ax.set_xlim(4.3, 7.7)
    # Canto superior direito: as linhas de cima ficam entre 4,7 e 5,6, entao a area
    # esta livre ali. No inferior direito a legenda cobria a linha do XGBoost.
    ax.legend(frameon=False, ncol=1, loc="upper right")
    ax.set_title("Cost of discarding history beyond five years", loc="left", fontsize=8.5)
    salva_fig(fig, "fig4_janela")

    # ---------- Figura 5: perfil sazonal ----------
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    meses = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax.plot(range(1, 13), perfil.values, "o-", color="#C44E52", lw=1.6, ms=5)
    ax2 = ax.twinx()
    ax2.plot(range(1, 13), temp.groupby(temp.index.month).tmin.mean().values, "s--",
             color="#4C72B0", lw=1.2, ms=4)
    ax2.set_ylabel("Mean minimum temperature (C)", color="#4C72B0")
    ax2.tick_params(axis="y", colors="#4C72B0")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(meses)
    ax.set_ylabel("Mean deaths per month", color="#C44E52")
    ax.tick_params(axis="y", colors="#C44E52")
    salva_fig(fig, "fig5_sazonal")

    # ---------- Figura 6: efeito da temperatura ----------
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    mods = ["sarima", "catboost", "xgboost"]
    y = np.arange(len(mods))[::-1]
    for i, m in enumerate(mods):
        d = V["tabela5"][m]
        ax.plot([d["ic_low"], d["ic_high"]], [y[i], y[i]], lw=2.4, color=COR[m])
        ax.plot(d["ganho"], y[i], "o", ms=6, color=COR[m], zorder=3)
    ax.axvline(0, color="#999999", lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([ROTULO[m] for m in mods])
    ax.set_xlabel("Reduction in sMAPE from the temperature covariate (pp)")
    salva_fig(fig, "fig6_temperatura")

    # ---------- JSON ----------
    (PAPER / "verified_numbers.json").write_text(
        json.dumps(V, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  json    paper/verified_numbers.json ({len(json.dumps(V))} bytes)")
    print("\nConferencia rapida:")
    print(f"  top-3 {' < '.join(ROTULO[m] for m in V['derivados']['ordem_top3'])}, "
          f"amplitude {V['derivados']['amplitude_top3']:.3f} pp contra "
          f"largura media {V['derivados']['largura_media_top3']:.3f} pp")
    print(f"  DM top-3: {V['derivados']['dm_top3_sig']}/{V['derivados']['dm_top3_total']} | "
          f"DM boosting vs top-3: {V['derivados']['dm_boosting_vs_top3_sig']}/{V['derivados']['dm_boosting_vs_top3_total']}"
          f" | todos os pares com boosting: {V['derivados']['dm_boosting_todos_pares_sig']}/{V['derivados']['dm_boosting_todos_pares_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
