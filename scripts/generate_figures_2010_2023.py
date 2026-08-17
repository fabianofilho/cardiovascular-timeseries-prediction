#!/usr/bin/env python3
"""Figuras da rodada 2010-2023 — dados 100% observados, sem síntese.

Substitui, para a série estendida, o generate_paper_figures.py (cujas figs 1 e 7
misturavam 24 meses sintéticos de 2019-2020 com dados reais). Aqui toda observação
vem da série extraída do SIM e versionada em results/series/.

Uso:
  python scripts/generate_figures_2010_2023.py \
    --series-csv results/series/serie_eventos_sp_sim_real_2010_2023.csv \
    --metrics-csv results/benchmark_sim_real_sp_2010_2023_metrics.csv \
    --predictions-csv results/benchmark_sim_real_sp_2010_2023_predictions.csv \
    --uncertainty-csv results/uncertainty_2010_2023_bootstrap_metrics.csv \
    --exog-metrics-csv results/benchmark_exog_temp_climatology_metrics.csv \
    --exog-observed-metrics-csv results/benchmark_exog_temp_observed_metrics.csv
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
IMG = BASE / "images"
IMG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.titlepad":     10,
})

# Paleta por entidade (cores fixas por modelo; variantes _temp herdam a cor base)
C = {
    "timesfm":  "#1565C0",
    "sarima":   "#E65100",
    "prophet":  "#2E7D32",
    "xgboost":  "#6A1B9A",
    "catboost": "#00838F",
    "obs":      "#212121",
}
LABELS = {
    "timesfm": "TimesFM", "sarima": "SARIMA", "prophet": "Prophet",
    "xgboost": "XGBoost", "catboost": "CatBoost",
}


def parse_args():
    p = argparse.ArgumentParser(description="Figuras da rodada 2010-2023")
    p.add_argument("--series-csv", default="results/series/serie_eventos_sp_sim_real_2010_2023.csv")
    p.add_argument("--metrics-csv", default="results/benchmark_sim_real_sp_2010_2023_metrics.csv")
    p.add_argument("--predictions-csv", default="results/benchmark_sim_real_sp_2010_2023_predictions.csv")
    p.add_argument("--uncertainty-csv", default="results/uncertainty_2010_2023_bootstrap_metrics.csv")
    p.add_argument("--exog-metrics-csv", default="")
    p.add_argument("--exog-observed-metrics-csv", default="")
    p.add_argument("--backtest-start", default="2015-01-01",
                   help="Data da primeira origem de teste (min_train_size=60)")
    p.add_argument("--suffix", default="_2010_2023", help="Sufixo dos arquivos de figura")
    return p.parse_args()


def base_model(name: str) -> str:
    return name.replace("_temp", "")


def fig1_time_series(obs, args):
    fig, ax = plt.subplots(figsize=(13, 4.5))

    for year in range(obs["date"].dt.year.min(), obs["date"].dt.year.max() + 1):
        ax.axvspan(pd.Timestamp(f"{year}-06-01"), pd.Timestamp(f"{year}-07-31"),
                   alpha=0.12, color="#1565C0", zorder=0)
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"),
               alpha=0.07, color="red", zorder=0)
    ax.text(pd.Timestamp("2020-08-01"), obs["value"].max() * 0.955, "COVID-19",
            color="#C62828", fontsize=9, ha="center", va="top", style="italic")

    start = pd.Timestamp(args.backtest_start)
    ax.axvline(start, color="gray", linestyle=":", linewidth=1.5)
    ax.text(start + pd.DateOffset(months=1), obs["value"].min() * 1.01,
            "← Training  |  Backtesting →", color="gray", fontsize=8.5, va="bottom")

    ax.plot(obs["date"], obs["value"], color=C["obs"], linewidth=1.6, zorder=2)
    ax.fill_between(obs["date"], obs["value"], obs["value"].min() - 200,
                    alpha=0.07, color=C["obs"])

    idx_max, idx_min = obs["value"].idxmax(), obs["value"].idxmin()
    ax.annotate(f"Max: {obs.loc[idx_max,'value']:,.0f}\n{obs.loc[idx_max,'date'].strftime('%b %Y')}",
                xy=(obs.loc[idx_max, "date"], obs.loc[idx_max, "value"]),
                xytext=(-78, -14), textcoords="offset points", fontsize=8.5, color="#1565C0",
                arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.9))
    ax.annotate(f"Min: {obs.loc[idx_min,'value']:,.0f}\n{obs.loc[idx_min,'date'].strftime('%b %Y')}",
                xy=(obs.loc[idx_min, "date"], obs.loc[idx_min, "value"]),
                xytext=(14, 22), textcoords="offset points", fontsize=8.5, color="#C62828",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.9))

    winter_patch = mpatches.Patch(color="#1565C0", alpha=0.25, label="Winter peak (Jun–Jul)")
    covid_patch = mpatches.Patch(color="red", alpha=0.15, label="COVID-19 period")
    ax.legend(handles=[winter_patch, covid_patch], loc="upper left", framealpha=0.9)

    ax.set_xlabel("Month")
    ax.set_ylabel("Cardiovascular deaths / month")
    ax.set_title(
        f"Monthly Cardiovascular Mortality — São Paulo State, Brazil "
        f"({obs['date'].min().strftime('%b %Y')} – {obs['date'].max().strftime('%b %Y')}) — observed SIM data only",
        fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(obs["date"].min(), obs["date"].max())

    fig.tight_layout()
    path = IMG / f"fig1_time_series{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig2_forecast_comparison(preds, metrics, args):
    models = [m for m in LABELS if m in set(preds["model"])]
    agg = (preds.groupby(["model", "date"])
           .agg(y_pred_mean=("y_pred", "mean"), y_pred_std=("y_pred", "std"),
                y_true=("y_true", "first"))
           .reset_index())

    fig, axes = plt.subplots(len(models), 1, figsize=(13, 3.4 * len(models)),
                             sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = agg[agg["model"] == model].sort_values("date")
        ax.fill_between(sub["date"], sub["y_pred_mean"] - sub["y_pred_std"],
                        sub["y_pred_mean"] + sub["y_pred_std"],
                        color=C[model], alpha=0.18, label="±1 SD")
        ax.plot(sub["date"], sub["y_true"], color=C["obs"], linewidth=2.0,
                label="Observed", zorder=3)
        ax.plot(sub["date"], sub["y_pred_mean"], color=C[model], linewidth=1.7,
                linestyle="--", label=f"{LABELS[model]} (mean)", zorder=2)

        m = metrics[metrics["model"] == model].iloc[0]
        ax.text(0.01, 0.96,
                f"MAE = {m['mae']:.0f}    RMSE = {m['rmse']:.0f}    sMAPE = {m['smape']:.2f}%",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                          edgecolor=C[model]))
        ax.set_ylabel("Deaths / month")
        ax.set_title(LABELS[model], fontweight="bold", color=C[model])
        ax.legend(loc="upper right", framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[-1].set_xlabel("Month")
    n_win = int(preds["window"].max())
    fig.suptitle(
        "Observed vs Predicted Cardiovascular Mortality\n"
        f"Rolling Origin Backtesting — São Paulo State ({n_win} windows, 2010–2023 series)",
        fontweight="bold", y=1.005)
    fig.tight_layout()
    path = IMG / f"fig2_forecast_comparison{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig3_metrics_ci(metrics, unc, preds, args):
    """sMAPE com IC95% de bootstrap por janela — a barra sem IC induziu o claim antigo."""
    sub = unc[unc["metric"] == "smape"].copy() if unc is not None else None
    models = [m for m in LABELS if m in set(metrics["model"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(models))
    for i, m in enumerate(models):
        point = float(metrics.loc[metrics["model"] == m, "smape"].values[0])
        color = C[base_model(m)]
        if sub is not None and m in set(sub["model"]):
            r = sub[sub["model"] == m].iloc[0]
            ax.errorbar(i, r["point"], yerr=[[r["point"] - r["ci_low"]], [r["ci_high"] - r["point"]]],
                        fmt="o", color=color, markersize=9, capsize=6, capthick=1.6,
                        elinewidth=1.6)
            ax.text(i + 0.08, r["point"], f"{r['point']:.2f}", fontsize=9.5, va="center")
        else:
            ax.plot(i, point, "o", color=color, markersize=9)
            ax.text(i + 0.08, point, f"{point:.2f}", fontsize=9.5, va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel("sMAPE (%)")
    n_win = int(preds["window"].max())
    ax.set_title(
        "sMAPE with 95% CI — bootstrap over whole rolling-origin windows\n"
        f"São Paulo State, 2010–2023 series ({n_win} windows × 6 horizons)",
        fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.tight_layout()
    path = IMG / f"fig3_smape_ci{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig4_smape_by_horizon(preds, args):
    preds = preds.copy()
    preds["smape_row"] = (200.0 * (preds["y_true"] - preds["y_pred"]).abs()
                          / (preds["y_true"].abs() + preds["y_pred"].abs()))
    hz = (preds.groupby(["model", "horizon"])["smape_row"]
          .agg(["mean", "std"]).reset_index())
    models = [m for m in LABELS if m in set(preds["model"])]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in models:
        sub = hz[hz["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub["mean"], color=C[base_model(model)], linewidth=2.0,
                marker="o", markersize=6, label=LABELS.get(model, model))

    n_h = int(preds["horizon"].max())
    ax.set_xlabel("Forecast Horizon (months ahead)")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("Forecast Accuracy by Horizon — São Paulo State (2010–2023 series)",
                 fontweight="bold")
    ax.set_xticks(range(1, n_h + 1))
    ax.set_xticklabels([f"h={i}" for i in range(1, n_h + 1)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = IMG / f"fig4_smape_by_horizon{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig7_seasonal_profile(obs, args):
    obs = obs.copy()
    obs["month"] = obs["date"].dt.month
    monthly = obs.groupby("month")["value"].agg(["mean", "std", "min", "max"]).reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def bar_color(m):
        if m in (6, 7):
            return "#1565C0"
        if m in (1, 2):
            return "#C62828"
        return "#78909C"

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bars = ax.bar(monthly["month"], monthly["mean"],
                  color=[bar_color(m) for m in monthly["month"]],
                  alpha=0.85, width=0.72, edgecolor="white", linewidth=0.9)
    ax.errorbar(monthly["month"], monthly["mean"], yerr=monthly["std"], fmt="none",
                ecolor="#37474F", elinewidth=1.4, capsize=5, capthick=1.4, alpha=0.7)
    ax.fill_between(monthly["month"], monthly["min"], monthly["max"],
                    alpha=0.08, color="#37474F", step="mid")
    for bar, mean_v in zip(bars, monthly["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + monthly["std"].max() * 0.05,
                f"{mean_v:.0f}", ha="center", va="bottom", fontsize=8.2)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Mean cardiovascular deaths / month")
    y0, y1 = obs["date"].dt.year.min(), obs["date"].dt.year.max()
    ax.set_title(
        f"Average Monthly Cardiovascular Mortality Profile\n"
        f"São Paulo State, Brazil ({y0}–{y1}, observed SIM data) — error bars = ±1 SD, shading = min–max",
        fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    handles = [mpatches.Patch(color="#1565C0", alpha=0.85, label="Winter peak (Jun–Jul)"),
               mpatches.Patch(color="#C62828", alpha=0.85, label="Summer trough (Jan–Feb)"),
               mpatches.Patch(color="#78909C", alpha=0.85, label="Other months")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = IMG / f"fig7_seasonal_profile{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig8_exog_effect(metrics, exog_clim, exog_obs, args):
    """Baseline vs temperatura-climatologia vs temperatura-observada (cenário-teto)."""
    scenarios = [("Baseline\n(sem exógena)", metrics, 1.0),
                 ("Temp. climatologia\n(sem vazamento)", exog_clim, 0.75)]
    if exog_obs is not None:
        scenarios.append(("Temp. observada\n(cenário-teto, vazamento rotulado)", exog_obs, 0.45))

    models = sorted({base_model(m) for df in (exog_clim,) for m in df["model"]})
    x = np.arange(len(models))
    width = 0.8 / len(scenarios)

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (label, df, alpha) in enumerate(scenarios):
        vals = []
        for m in models:
            row = df[df["model"].apply(base_model) == m]
            vals.append(float(row["smape"].values[0]) if len(row) else np.nan)
        bars = ax.bar(x + (j - (len(scenarios) - 1) / 2) * width, vals, width * 0.92,
                      label=label, color=[C[m] for m in models], alpha=alpha,
                      edgecolor="white", linewidth=0.9)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("Effect of Temperature Exogenous Variable by Policy\n"
                 "Same rolling-origin windows; opacity encodes scenario",
                 fontweight="bold")
    ax.legend(loc="upper center", framealpha=0.9, fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.tight_layout()
    path = IMG / f"fig8_exog_effect{args.suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def main():
    args = parse_args()
    obs = pd.read_csv(BASE / args.series_csv, parse_dates=["date"])
    metrics = pd.read_csv(BASE / args.metrics_csv)
    preds = pd.read_csv(BASE / args.predictions_csv, parse_dates=["date"])
    if "horizon" not in preds.columns:
        raise SystemExit("predictions.csv sem coluna horizon — rode o run_benchmark.py atual.")
    unc = None
    unc_path = BASE / args.uncertainty_csv
    if unc_path.exists():
        unc = pd.read_csv(unc_path)

    print(f"\nGerando figuras ({len(obs)} meses observados) → {IMG}\n")
    fig1_time_series(obs, args)
    fig2_forecast_comparison(preds, metrics, args)
    fig3_metrics_ci(metrics, unc, preds, args)
    fig4_smape_by_horizon(preds, args)
    fig7_seasonal_profile(obs, args)

    if args.exog_metrics_csv and (BASE / args.exog_metrics_csv).exists():
        exog_clim = pd.read_csv(BASE / args.exog_metrics_csv)
        exog_obs = None
        if args.exog_observed_metrics_csv and (BASE / args.exog_observed_metrics_csv).exists():
            exog_obs = pd.read_csv(BASE / args.exog_observed_metrics_csv)
        fig8_exog_effect(metrics, exog_clim, exog_obs, args)

    print("\nConcluído.\n")


if __name__ == "__main__":
    main()
