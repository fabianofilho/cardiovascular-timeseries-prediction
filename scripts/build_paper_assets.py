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

import numpy as np
import pandas as pd
from cv_timeseries.evaluate import mase_denominador
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
    "naive": "Naive", "snaive": "Seasonal naive",
    "snaive_drift": "Seasonal naive + drift",
}
# HEX sem '#': entram em \definecolor{...}{HTML}{...} no preambulo
COR = {
    "prophet": "4C72B0", "sarima": "DD8452", "timesfm": "55A868",
    "xgboost": "C44E52", "catboost": "8172B3",
}
TOP3 = ["prophet", "sarima", "timesfm"]
ORDEM = ["prophet", "sarima", "timesfm", "catboost", "xgboost"]
# Referencias ingenuas. Ficam separadas de ORDEM porque nao competem: elas sao a
# barra que os modelos precisam passar, e aparecem num bloco proprio da Tabela 1.
BASELINES = ["snaive_drift", "snaive", "naive"]



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

def coords(pares):
    return " ".join(f"({x},{y})" for x, y in pares)


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


GERADAS: list[str] = []


def escreve_figura(nome, corpo):
    """Grava um tikzpicture que o manuscrito chama por \\input{figures/<nome>}."""
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"{nome}.tex").write_text(AVISO + corpo, encoding="utf-8")
    GERADAS.append(nome)
    print(f"  figura  paper/figures/{nome}.tex")


def defs_cor():
    """Paleta para o preambulo, definida uma vez em vez de por figura."""
    return "\n".join(f"\\definecolor{{c{k}}}{{HTML}}{{{v}}}" for k, v in COR.items())


def rasteriza_figuras(dpi=300):
    """Compila cada figura isolada e salva o PNG irmao.

    O PNG nao entra no documento: ele existe para ser aberto e conferido, para ser o
    que se entrega a quem pediu "as figuras", porque PNG abre em qualquer lugar e .tex
    de pgfplots so vira imagem depois de compilar, e para ser embutido no .docx de
    revisao. 300 dpi porque esse ultimo uso e impressao, e e o unico dos tres que tem
    exigencia de resolucao: o menor denominador comum mandaria no conjunto todo.

    Enquanto as figuras eram matplotlib isto saia de graca no savefig. Com figura em
    codigo, o unico jeito de ter o PNG e compilar e rasterizar, entao o passo tem que
    ser explicito: sem ele o gerador para de emitir PNG e ninguem percebe.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("tectonic") is None:
        print("  [AVISO] tectonic ausente: PNG das figuras nao gerado")
        return
    try:
        import pymupdf
    except ImportError:
        print("  [AVISO] pymupdf ausente: PNG das figuras nao gerado")
        return

    preambulo = ("\\documentclass[11pt,border=2pt]{standalone}\n"
                 "\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
                 "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n"
                 "\\usepackage{xcolor}\n" + defs_cor() + "\n\\begin{document}\n")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for nome in GERADAS:
            corpo = (FIG / f"{nome}.tex").read_text(encoding="utf-8")
            (td / f"{nome}.tex").write_text(preambulo + corpo + "\n\\end{document}\n",
                                            encoding="utf-8")
            r = subprocess.run(["tectonic", "-X", "compile", f"{nome}.tex"],
                               cwd=td, capture_output=True, text=True)
            if r.returncode != 0 or not (td / f"{nome}.pdf").exists():
                print(f"  [AVISO] {nome}: falhou ao compilar isolada, sem PNG")
                continue
            doc = pymupdf.open(td / f"{nome}.pdf")
            doc[0].get_pixmap(dpi=dpi).save(FIG / f"{nome}.png")
            doc.close()
            kb = (FIG / f"{nome}.png").stat().st_size / 1024
            print(f"  png     paper/figures/{nome}.png ({kb:.0f} KB, {dpi} dpi)")


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
    hor = pd.read_csv(RES / "uncertainty_2010_2023_error_by_horizon.csv")
    base_ing = pd.read_csv(RES / "benchmark_baselines_2010_2023_predictions.csv",
                           parse_dates=["date"])

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
    # Baselines ingenuas nas MESMAS janelas, para o bootstrap ser pareado com elas.
    for m in BASELINES:
        yt, yp = matriz(base_ing, m)
        sm[m] = smape_vec(yt, yp)
        ae[m] = np.abs(yp - yt)
        se[m] = (yp - yt) ** 2
    boot = bootstrap_janelas(sm, nw)

    # Denominador do MASE: MAE do naive sazonal EM AMOSTRA sobre a serie inteira.
    # Constante de escala, para o MASE ser comparavel entre modelos.
    den_mase = mase_denominador(serie, m=12)
    V["mase_denominador"] = float(den_mase)

    TODAS = ORDEM + BASELINES
    V["tabela1"] = {}
    for m in TODAS:
        lo, hi = ic(boot[m])
        V["tabela1"][m] = {
            "mae": float(ae[m].mean()), "rmse": float(np.sqrt(se[m].mean())),
            "smape": float(sm[m].mean()), "ic_low": lo, "ic_high": hi, "ic_width": hi - lo,
            "mase": float(ae[m].mean() / den_mase),
        }
    # O melhor valor por coluna considera SO os modelos, nao as referencias: negrito numa
    # baseline leria como se ela fosse concorrente, e ela e a regua.
    melhor = {k: min(ORDEM, key=lambda m: V["tabela1"][m][k])
              for k in ("mae", "rmse", "smape", "mase")}

    def linha_tab1(m):
        d = V["tabela1"][m]
        cel = []
        for k, casas in (("mae", 1), ("rmse", 1), ("smape", 2), ("mase", 3)):
            s = num(d[k], casas)
            cel.append(f"\\textbf{{{s}}}" if melhor[k] == m else s)
        cel.append(f"$[{d['ic_low']:.2f}, {d['ic_high']:.2f}]$")
        return f"{ROTULO[m]} & " + " & ".join(cel) + " \\\\"

    linhas = [linha_tab1(m) for m in ORDEM]
    linhas.append("\\midrule")
    linhas += [linha_tab1(m) for m in BASELINES]
    escreve_tabela("tab1_desempenho", f"""\\begin{{table}}[htbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\caption{{Forecasting accuracy for monthly cardiovascular mortality in the state of
S\\~ao Paulo, Brazil, 2010--2023. All models were evaluated on the same
{nw * nh} out-of-sample forecasts from {nw} rolling origin windows with a
{nh}-month horizon.}}
\\label{{tab:desempenho}}
\\begin{{tabular}}{{lrrrrc}}
\\toprule
Model & MAE & RMSE & sMAPE (\\%) & MASE & 95\\% CI of sMAPE \\\\
\\midrule
{chr(10).join(linhas)}
\\bottomrule
\\end{{tabular}}

\\vspace{{0.5em}}
\\begin{{minipage}}{{\\textwidth}}
\\footnotesize
Bold marks the best value in each metric column among the forecasting models; the three
rows below the rule are naive references, not competitors. MASE is the MAE divided by the
in-sample MAE of the seasonal naive method ({den_mase:.1f} deaths), so MASE below 1 beats
repeating the same month of the previous year and MASE above 1 loses to it. MAE and RMSE
are in deaths per month;
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

    # ---------- figuras, em pgfplots ----------
    # Figura e codigo, nao binario: cada uma sai como tikzpicture em
    # paper/figures/figN_nome.tex e entra no manuscrito por \input pelo nome,
    # igual as tabelas. Assim o manuscrito cabe num .tex unico e a figura herda
    # a fonte do documento em vez de carregar a do matplotlib.
    # ---------- fig1: serie ----------
    pts = coords([(f"{d.year + (d.month - 1) / 12:.4f}", f"{v:.0f}") for d, v in serie.items()])
    escreve_figura("fig1_serie", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.95\textwidth, height=5.0cm,
  xlabel={{}}, ylabel={{Deaths per month}},
  xmin=2009.8, xmax=2024.2, ymin=5400, ymax=11100,
  xtick={{2010,2012,2014,2016,2018,2020,2022,2024}},
  xticklabel style={{/pgf/number format/1000 sep=}},
  % sem isto o pgfplots troca o eixo por \cdot 10^4, que e ilegivel aqui
  scaled y ticks=false,
  ytick={{6000,7000,8000,9000,10000,11000}},
  yticklabel style={{/pgf/number format/fixed, /pgf/number format/1000 sep={{,}}}},
  axis lines=left, tick align=outside, tick pos=left,
  every axis plot/.append style={{line width=0.5pt}},
]
\addplot[draw=black!80, mark=none] coordinates {{{pts}}};
\addplot[draw=none, fill=cprophet, fill opacity=0.10, forget plot]
  coordinates {{(2021.0,5400) (2024.0,5400) (2024.0,11100) (2021.0,11100)}} \closedcycle;
% ancorado a direita e dentro do limite do eixo: centralizado em 2022.5 o rotulo
% estourava a borda e saia cortado
\node[anchor=east, font=\scriptsize, text=cprophet!80!black]
  at (axis cs:2023.9,10500) {{test window shared with the earlier round}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig2: sMAPE com IC ----------
    linhas = []
    for i, m in enumerate(ORDEM):
        y = len(ORDEM) - i
        d = V["tabela1"][m]
        linhas.append(
            f"\\addplot[draw=c{m}, line width=2.0pt, mark=none] "
            f"coordinates {{({d['ic_low']:.4f},{y}) ({d['ic_high']:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.2pt, draw=c{m}, fill=c{m}] "
            f"coordinates {{({d['smape']:.4f},{y})}};")
    p3 = [V["tabela1"][m]["smape"] for m in TOP3]
    amp, larg = V["derivados"]["amplitude_top3"], V["derivados"]["largura_media_top3"]
    ticks = ",".join(str(len(ORDEM) - i) for i in range(len(ORDEM)))
    labs = ",".join(ROTULO[m] for m in ORDEM)
    escreve_figura("fig2_smape_ic", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.2cm,
  xlabel={{sMAPE (\%)}}, xmin=4.0, xmax=7.8,
  xtick={{4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1,
                     /pgf/number format/zerofill}},
  ymin=0.4, ymax=5.6, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Spread among the leading three: {amp:.2f} pp; mean interval width: {larg:.2f} pp}},
]
\addplot[draw=none, fill=black, fill opacity=0.14, forget plot]
  coordinates {{({min(p3):.4f},0.4) ({max(p3):.4f},0.4) ({max(p3):.4f},5.6) ({min(p3):.4f},5.6)}}
  \closedcycle;
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig3: erro por horizonte ----------
    series_h, leg = [], []
    for m in ORDEM:
        g = hor[hor.model == m].sort_values("horizon")
        # mark options e obrigatorio: `mark=*` NAO herda o `draw` da serie, e sem isto
        # o ponto sai preto sobre a linha colorida. So aparece ampliando o PDF.
        series_h.append(f"\\addplot[draw=c{m}, mark=*, mark size=1.6pt, line width=0.9pt, "
                        f"mark options={{draw=c{m}, fill=c{m}}}] "
                        f"coordinates {{{coords(zip(g.horizon, [f'{v:.4f}' for v in g.smape]))}}};")
        leg.append(ROTULO[m])
    escreve_figura("fig3_horizonte", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.2cm,
  xlabel={{Forecast horizon (months)}}, ylabel={{sMAPE (\%)}},
  xmin=0.7, xmax=6.3, xtick={{1,2,3,4,5,6}},
  axis lines=left, tick align=outside, tick pos=left,
  legend style={{at={{(0.02,0.98)}}, anchor=north west, draw=none, fill=none,
                 font=\scriptsize, legend columns=3, column sep=4pt}},
  ymax=8.6,
]
{chr(10).join(series_h)}
\legend{{{','.join(leg)}}}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig4: expanding vs deslizante ----------
    linhas = []
    for i, m in enumerate(ORDEM):
        y = len(ORDEM) - i
        e, s = V["tabela4"][m]["expanding"], V["tabela4"][m]["deslizante"]
        dif = s - e
        cor = "C44E52" if dif > 0 else "55A868"
        linhas.append(
            f"\\addplot[draw=black!28, line width=1.2pt, mark=none] "
            f"coordinates {{({e:.4f},{y}) ({s:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw=cprophet, fill=cprophet] "
            f"coordinates {{({e:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, draw=csarima, fill=csarima] "
            f"coordinates {{({s:.4f},{y})}};\n"
            f"\\definecolor{{d{i}}}{{HTML}}{{{cor}}}\n"
            f"\\node[anchor=west, font=\\scriptsize, text=d{i}] "
            f"at (axis cs:{max(e, s) + 0.10:.4f},{y}) {{{dif:+.2f} pp}};")
    ticks = ",".join(str(len(ORDEM) - i) for i in range(len(ORDEM)))
    labs = ",".join(ROTULO[m] for m in ORDEM)
    escreve_figura("fig4_janela", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=5.4cm,
  xlabel={{sMAPE (\%)}}, xmin=4.3, xmax=7.7,
  xtick={{4.5,5.0,5.5,6.0,6.5,7.0,7.5}},
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1,
                     /pgf/number format/zerofill}},
  ymin=0.4, ymax=5.7, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
  title style={{align=left, font=\small}},
  title={{Cost of discarding history beyond five years}},
]
{chr(10).join(linhas)}
\node[anchor=east, font=\scriptsize] at (axis cs:7.65,5.35)
  {{\textcolor{{cprophet}}{{$\bullet$}} Expanding \quad
    \textcolor{{csarima}}{{$\bullet$}} Sliding 60}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig5: sazonal com eixo duplo ----------
    perfil = serie.groupby(serie.index.month).mean()
    tmin = temp.groupby(temp.index.month).tmin.mean()
    cm = coords(zip(range(1, 13), [f"{v:.1f}" for v in perfil]))
    ct = coords(zip(range(1, 13), [f"{v:.2f}" for v in tmin]))
    meses = "J,F,M,A,M,J,J,A,S,O,N,D"
    escreve_figura("fig5_sazonal", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.80\textwidth, height=4.8cm,
  axis y line*=left, axis x line*=bottom,
  xmin=0.5, xmax=12.5, xtick={{1,...,12}}, xticklabels={{{meses}}},
  ylabel={{Mean deaths per month}}, ylabel style={{text=cxgboost}},
  scaled y ticks=false, ytick={{6500,7000,7500,8000}},
  yticklabel style={{text=cxgboost, /pgf/number format/fixed,
                     /pgf/number format/1000 sep={{,}}}},
  tick align=outside,
]
\addplot[draw=cxgboost, mark=*, mark size=2.0pt, line width=1.1pt]
  coordinates {{{cm}}};
\end{{axis}}
\begin{{axis}}[
  width=0.80\textwidth, height=4.8cm,
  axis y line*=right, axis x line=none,
  xmin=0.5, xmax=12.5,
  ylabel={{Mean minimum temperature (C)}}, ylabel style={{text=cprophet}},
  yticklabel style={{text=cprophet}},
  tick align=outside,
]
\addplot[draw=cprophet, mark=square*, mark size=1.7pt, line width=0.9pt, dashed]
  coordinates {{{ct}}};
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- fig6: efeito da temperatura ----------
    mods = ["sarima", "catboost", "xgboost"]
    linhas = []
    for i, m in enumerate(mods):
        y = len(mods) - i
        d = V["tabela5"][m]
        linhas.append(
            f"\\addplot[draw=c{m}, line width=2.0pt, mark=none] "
            f"coordinates {{({d['ic_low']:.4f},{y}) ({d['ic_high']:.4f},{y})}};\n"
            f"\\addplot[only marks, mark=*, mark size=2.2pt, draw=c{m}, fill=c{m}] "
            f"coordinates {{({d['ganho']:.4f},{y})}};")
    ticks = ",".join(str(len(mods) - i) for i in range(len(mods)))
    labs = ",".join(ROTULO[m] for m in mods)
    escreve_figura("fig6_temperatura", rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.78\textwidth, height=3.8cm,
  xlabel={{Reduction in sMAPE from the temperature covariate (pp)}},
  xmin=-0.06, xmax=0.62,
  % com valores pequenos o pgfplots gera tick automatico em notacao cientifica
  % (-5 \cdot 10^{-2}) e os rotulos colidem. Tick e formato explicitos.
  xtick={{0,0.1,0.2,0.3,0.4,0.5,0.6}},
  scaled x ticks=false,
  xticklabel style={{/pgf/number format/fixed, /pgf/number format/precision=1}},
  ymin=0.4, ymax=3.6, ytick={{{ticks}}}, yticklabels={{{labs}}},
  axis lines=left, tick align=outside, tick pos=left,
]
\draw[black!45, dashed, line width=0.6pt]
  (axis cs:0,0.4) -- (axis cs:0,3.6);
{chr(10).join(linhas)}
\end{{axis}}
\end{{tikzpicture}}""")

    # ---------- JSON ----------
    (PAPER / "verified_numbers.json").write_text(
        json.dumps(V, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  json    paper/verified_numbers.json ({len(json.dumps(V))} bytes)")

    # PNG irmao de cada figura: nao entra no documento, serve para conferir com o Read
    # e e a forma em que a figura se entrega a quem pede "as figuras".
    rasteriza_figuras()
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
