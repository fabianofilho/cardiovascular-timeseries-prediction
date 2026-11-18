"""Monta as tabelas da v1b a partir das tabelas geradas da v1.

O miolo numerico (o bloco tabular) sai de paper/tables, gerado por
scripts/build_paper_assets.py, e nao e tocado aqui alem de cortar linhas e trocar rotulos de
coluna. So a legenda e a nota de rodape sao reescritas. Rodar de dentro de paper_v1b/:

    python build_tables.py
"""
from __future__ import annotations

import re
from pathlib import Path

ORIGEM = Path("../paper/tables")
DESTINO = Path("tables")

TABELAS = {
    # destino: (origem, label, caption, nota, linhas a cortar, trocas no cabecalho)
    "tab1_desempenho": (
        "tab1_desempenho", "tab:desempenho",
        "Accuracy of five forecasting models and three naive references for monthly "
        "cardiovascular deaths in S\\~ao Paulo state, 618 out-of-sample forecasts from 103 "
        "rolling origin windows with a 6-month horizon.",
        "MAE and RMSE in deaths per month; sMAPE, symmetric mean absolute percentage error. "
        "MASE divides MAE by the in-sample one-step MAE of the seasonal naive method (412.9 "
        "deaths); since the scored forecasts are out of sample and up to six steps ahead, the "
        "seasonal naive row (1.148) is the benchmark, not the value 1. Bold, best value among "
        "the five models. The 95\\% CI is a percentile bootstrap over whole windows "
        "($B=10{,}000$), and Width is its span in percentage points. The interval describes "
        "uncertainty about the metric, not about an individual forecast.",
        [], {},
    ),
    "tab2_pares": (
        "tab2_pares", "tab:pares",
        "Paired differences in sMAPE among the three most accurate models. A positive "
        "difference means the first model has the larger error.",
        "pp, percentage points. Bootstrap differences are computed within each replicate on "
        "the same resampled windows. DM, Diebold-Mariano test with the Harvey-Leybourne-Newbold "
        "correction, run separately at each of the six horizons; the column counts horizons "
        "with $p<0.05$.",
        [], {},
    ),
    "tab3_serie": (
        "tab3_serie", "tab:serie",
        "sMAPE on the same 36 target months (January 2021 to December 2023) with short and "
        "long training history.",
        "The short-history column comes from a run on 2019--2023 data, in which each window "
        "had 24 to 54 months of training ($n=186$ forecasts per model); the long-history "
        "column is the subset of the main analysis that falls on the same target months "
        "($n=201$). Only the amount of training data differs.",
        [], {},
    ),
    "tab4_janela": (
        "tab4_janela", "tab:janela",
        "sMAPE with an expanding training window and with a window limited to the most "
        "recent 60 months, on identical test origins. A negative difference means the "
        "expanding window is more accurate.",
        "The expanding window uses 60 to 162 months of training; the sliding window uses the "
        "last 60. The bootstrap is paired by window. DM as in Table~\\ref{tab:pares}.",
        ["enriched"], {},
    ),
    "tab5_temperatura": (
        "tab5_temperatura", "tab:temperatura",
        "sMAPE with and without monthly minimum temperature as a covariate. TimesFM has no "
        "covariate interface in the version used and is not included.",
        "With temp.: the future covariate is the month-of-year mean of temperature computed "
        "from data up to each window's training end, so no future value is used. Ceiling: "
        "the observed future temperature, which no real forecast could use; it bounds what a "
        "perfect weather forecast could add. Gain is positive when temperature reduces error. "
        "DM as in Table~\\ref{tab:pares}.",
        [], {},
    ),
    "tab6_calibracao": (
        "tab6_calibracao", "tab:calibracao",
        "Empirical coverage and interval score of the nominal 95\\% prediction intervals of "
        "SARIMA and Prophet, by forecast horizon, over the 103 windows.",
        "Coverage is the fraction of observed values inside the interval (target 0.950). "
        "MPIW, mean interval width in deaths per month. Interval score as in "
        "\\citet{gneiting2007}: width plus a penalty for each observation outside the "
        "interval, proportional to the miss; lower is better.",
        [], {},
    ),
    # Suplemento
    "s1_variantes": (
        "tab7_variantes", "tab:variantes",
        "Feature engineering variants for the two gradient boosting models, under the "
        "protocol of Table~\\ref{tab:desempenho}. The last two columns compare each CatBoost "
        "variant with the seasonal naive method.",
        "$\\Delta$, CatBoost minus seasonal naive sMAPE, with a paired bootstrap 95\\% CI. DM, "
        "horizons with $p<0.05$ out of six. Direct multi-step fits one model per horizon.",
        [], {},
    ),
    "s2_optuna": (
        "revisao_tab_optuna", "tab:optuna",
        "Bayesian hyperparameter search for the two gradient boosting models.",
        "Training-only: 40 trials with the search restricted to the first 60 months, so no "
        "test window enters it. Test-exposed: 60 trials allowed to score on the test windows, "
        "an upper bound that a real procedure cannot reach. Gain is positive when the search "
        "improved on the untuned configuration of Table~\\ref{tab:desempenho}.",
        [], {"Honest": "Training-only", "Leaked": "Test-exposed"},
    ),
    "s3_covid": (
        "tab8_covid", "tab:covid",
        "sMAPE of the three most accurate models with and without the COVID-19 period in the "
        "test set.",
        "Windows whose test months overlap the excluded range are dropped whole, so the "
        "pairing is kept; no model is refitted. Pairs separated: paired comparisons among the "
        "three whose interval excludes zero and whose DM test reaches $p<0.05$ in at least "
        "three horizons. Worst DM: the largest count of significant horizons in any pair.",
        [], {},
    ),
    "s4_temp_calendario": (
        "revisao_tab_temp_diffcal", "tab:tempdiffcal",
        "Temperature gain for the boosting models with and without explicit calendar terms "
        "and seasonal differencing.",
        "Gain in percentage points, positive when temperature reduces error. The lags-only "
        "rows reproduce Table~\\ref{tab:temperatura}.",
        [], {},
    ),
}


def tabular(texto: str) -> tuple[str, str]:
    """Devolve o bloco tabular e o \\setlength de tabcolsep, se houver."""
    bloco = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", texto, re.S).group(0)
    sep = re.search(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", texto)
    return bloco, sep.group(0) if sep else ""


def cortar(bloco: str, termos: list[str]) -> str:
    linhas = bloco.splitlines()
    fica = [ln for ln in linhas if not any(t in ln for t in termos)]
    # um \midrule que ficou colado no \bottomrule depois do corte sai junto
    saida = []
    for i, ln in enumerate(fica):
        prox = fica[i + 1].strip() if i + 1 < len(fica) else ""
        if ln.strip() == "\\midrule" and prox == "\\bottomrule":
            continue
        saida.append(ln)
    return "\n".join(saida)


def main() -> None:
    DESTINO.mkdir(exist_ok=True)
    for destino, (origem, label, caption, nota, corte, trocas) in TABELAS.items():
        bruto = (ORIGEM / f"{origem}.tex").read_text(encoding="utf-8")
        bloco, sep = tabular(bruto)
        if corte:
            bloco = cortar(bloco, corte)
        for velho, novo in trocas.items():
            bloco = bloco.replace(velho, novo)
        saida = (
            f"% gerado por paper_v1b/build_tables.py a partir de paper/tables/{origem}.tex"
            " -- nao editar a mao\n"
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            + (sep + "\n" if sep else "")
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n{bloco}\n\n"
            "\\vspace{0.5em}\n\\begin{minipage}{\\textwidth}\n\\footnotesize\n"
            f"{nota}\n\\end{{minipage}}\n\\end{{table}}\n"
        )
        (DESTINO / f"{destino}.tex").write_text(saida, encoding="utf-8")
        print(f"{destino}.tex <- {origem}.tex")


if __name__ == "__main__":
    main()
