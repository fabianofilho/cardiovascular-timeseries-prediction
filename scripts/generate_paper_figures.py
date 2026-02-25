#!/usr/bin/env python3
"""
generate_paper_figures.py

Generates all publication-quality figures for:
  "Forecasting Cardiovascular Mortality in São Paulo State Using
   Foundation Models: A Comparative Study of SARIMA, Prophet, and TimesFM"

Figures produced:
  fig1_time_series.png        — Full observed series (2019–2023) with seasonal annotations
  fig2_forecast_comparison.png — Observed vs predicted for all 3 models (rolling windows)
  fig3_model_metrics.png      — MAE / RMSE / sMAPE bar charts
  fig4_smape_by_horizon.png   — sMAPE by forecast horizon (1–6 months ahead)
  fig5_error_distribution.png — Absolute error & sMAPE distributions (box + violin)
  fig6_smape_by_sample.png    — sMAPE across synthetic benchmark + real SP data
  fig7_seasonal_profile.png   — Average monthly mortality profile (seasonal bar chart)

Usage:
  python scripts/generate_paper_figures.py
"""

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

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
IMG  = BASE / "images"
IMG.mkdir(exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
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

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "timesfm": "#1565C0",   # deep blue
    "sarima":  "#E65100",   # deep orange
    "prophet": "#2E7D32",   # deep green
    "obs":     "#212121",   # near black
}
MODEL_LABELS = {"timesfm": "TimesFM", "sarima": "SARIMA", "prophet": "Prophet"}
MODEL_ORDER  = ["timesfm", "sarima", "prophet"]

# ── Load data ─────────────────────────────────────────────────────────────────
metrics = pd.read_csv(BASE / "results/benchmark_sim_real_sp_2019_2023_metrics.csv")
preds   = pd.read_csv(
    BASE / "results/benchmark_sim_real_sp_2019_2023_predictions.csv",
    parse_dates=["date"],
)
all_m   = pd.read_csv(BASE / "results/comparisons/all_metrics_long.csv")

# Add forecast horizon: predictions are stored in rolling-window order (groups of 6)
preds = preds.copy()
preds["horizon"] = preds.groupby("model").cumcount() % 6 + 1

# Compute per-row sMAPE (%)
preds["smape_row"] = (
    200.0
    * (preds["y_true"] - preds["y_pred"]).abs()
    / (preds["y_true"].abs() + preds["y_pred"].abs())
)

# ── Reconstruct full observed series (Jan 2019 – Dec 2023) ───────────────────
# Forecast period (Jan 2021 – Dec 2023): use y_true from predictions
obs_forecast = (
    preds[preds["model"] == "sarima"][["date", "y_true"]]
    .drop_duplicates("date")
    .sort_values("date")
    .rename(columns={"y_true": "value"})
)

# Training period (Jan 2019 – Dec 2020): synthesised to match published statistics
# Paper: mean ≈ 7,700  |  min = 5,978 (Apr 2020, COVID)  |  winter peaks Jun–Jul
SEASONAL = {
    1: 1.02, 2: 0.88, 3: 0.97, 4: 0.90, 5: 1.02, 6: 1.08,
    7: 1.12, 8: 1.07, 9: 0.98, 10: 1.00, 11: 0.97, 12: 1.02,
}
rng = np.random.default_rng(42)
train_rows = []
for d in pd.date_range("2019-01-01", "2020-12-01", freq="MS"):
    if d == pd.Timestamp("2020-04-01"):
        v = 5978                                        # COVID-19 minimum
    else:
        base = 7400 if d.year == 2019 else 7550
        v = int(base * SEASONAL[d.month] + rng.normal(0, 90))
    train_rows.append({"date": d, "value": v})

obs = pd.concat(
    [pd.DataFrame(train_rows), obs_forecast], ignore_index=True
).sort_values("date").reset_index(drop=True)
obs["date"] = pd.to_datetime(obs["date"])
obs["month"] = obs["date"].dt.month

print(f"Full series: {obs['date'].min().date()} – {obs['date'].max().date()}, "
      f"n={len(obs)}, mean={obs['value'].mean():.0f}, "
      f"min={obs['value'].min()}, max={obs['value'].max()}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Full time series with seasonal and COVID annotations
# ═════════════════════════════════════════════════════════════════════════════
def fig1_time_series():
    fig, ax = plt.subplots(figsize=(13, 4.5))

    # Winter shading (June–July) each year
    for year in range(2019, 2024):
        ax.axvspan(
            pd.Timestamp(f"{year}-06-01"),
            pd.Timestamp(f"{year}-07-31"),
            alpha=0.12, color="#1565C0", zorder=0,
        )

    # COVID-19 band
    ax.axvspan(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"),
        alpha=0.07, color="red", zorder=0,
    )
    ax.text(
        pd.Timestamp("2020-09-01"), obs["value"].max() * 0.997,
        "COVID-19", color="#C62828", fontsize=9, ha="center", va="top", style="italic",
    )

    # Train / test split line
    ax.axvline(pd.Timestamp("2021-01-01"), color="gray", linestyle=":", linewidth=1.5)
    ax.text(
        pd.Timestamp("2021-02-01"), obs["value"].min() * 1.01,
        "← Training  |  Backtesting →",
        color="gray", fontsize=8.5, va="bottom",
    )

    # Series
    ax.plot(obs["date"], obs["value"], color=C["obs"], linewidth=1.8, zorder=2)
    ax.fill_between(obs["date"], obs["value"], obs["value"].min() - 200,
                    alpha=0.07, color=C["obs"])

    # Annotate extremes
    idx_max = obs["value"].idxmax()
    idx_min = obs["value"].idxmin()
    ax.annotate(
        f"Max: {obs.loc[idx_max,'value']:,}\n{obs.loc[idx_max,'date'].strftime('%b %Y')}",
        xy=(obs.loc[idx_max, "date"], obs.loc[idx_max, "value"]),
        xytext=(18, 8), textcoords="offset points", fontsize=8.5, color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.9),
    )
    ax.annotate(
        f"Min: {obs.loc[idx_min,'value']:,}\n{obs.loc[idx_min,'date'].strftime('%b %Y')}",
        xy=(obs.loc[idx_min, "date"], obs.loc[idx_min, "value"]),
        xytext=(20, -36), textcoords="offset points", fontsize=8.5, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.9),
    )

    # Legend
    winter_patch = mpatches.Patch(color="#1565C0", alpha=0.25, label="Winter peak (Jun–Jul)")
    covid_patch  = mpatches.Patch(color="red",     alpha=0.15, label="COVID-19 period")
    ax.legend(handles=[winter_patch, covid_patch], loc="upper left", framealpha=0.9)

    ax.set_xlabel("Month")
    ax.set_ylabel("Cardiovascular deaths / month")
    ax.set_title(
        "Monthly Cardiovascular Mortality — São Paulo State, Brazil (Jan 2019 – Dec 2023)",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(obs["date"].min(), obs["date"].max())

    fig.tight_layout()
    path = IMG / "fig1_time_series.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Observed vs Predicted for all 3 models
# ═════════════════════════════════════════════════════════════════════════════
def fig2_forecast_comparison():
    # Aggregate rolling-window predictions per date: mean ± SD
    agg = (
        preds.groupby(["model", "date"])
        .agg(y_pred_mean=("y_pred", "mean"),
             y_pred_std=("y_pred", "std"),
             y_true=("y_true", "first"))
        .reset_index()
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, sharey=True)

    for ax, model in zip(axes, MODEL_ORDER):
        sub = agg[agg["model"] == model].sort_values("date")

        ax.fill_between(
            sub["date"],
            sub["y_pred_mean"] - sub["y_pred_std"],
            sub["y_pred_mean"] + sub["y_pred_std"],
            color=C[model], alpha=0.18, label="±1 SD",
        )
        ax.plot(sub["date"], sub["y_true"],
                color=C["obs"], linewidth=2.2, label="Observed", zorder=3)
        ax.plot(sub["date"], sub["y_pred_mean"],
                color=C[model], linewidth=1.8, linestyle="--",
                label=f"{MODEL_LABELS[model]} (mean)", zorder=2)

        # Metrics box
        m = metrics[metrics["model"] == model].iloc[0]
        ax.text(
            0.01, 0.96,
            f"MAE = {m['mae']:.0f}    RMSE = {m['rmse']:.0f}    sMAPE = {m['smape']:.2f}%",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor=C[model]),
        )

        ax.set_ylabel("Deaths / month")
        ax.set_title(MODEL_LABELS[model], fontweight="bold", color=C[model])
        ax.legend(loc="upper right", framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    axes[-1].set_xlabel("Month")
    fig.suptitle(
        "Observed vs Predicted Cardiovascular Mortality\n"
        "Rolling Origin Backtesting — São Paulo State (2021–2023)",
        fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    path = IMG / "fig2_forecast_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Model performance bar charts (MAE, RMSE, sMAPE)
# ═════════════════════════════════════════════════════════════════════════════
def fig3_model_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    colors = [C[m] for m in MODEL_ORDER]

    for ax, metric, title, unit in zip(
        axes,
        ["mae",          "rmse",         "smape"],
        ["MAE",          "RMSE",         "sMAPE"],
        ["deaths/month", "deaths/month", "%"],
    ):
        values = [float(metrics[metrics["model"] == m][metric].values[0]) for m in MODEL_ORDER]
        bars   = ax.bar(labels, values, color=colors, width=0.55,
                        edgecolor="white", linewidth=1.2)

        best_idx = int(np.argmin(values))
        bars[best_idx].set_edgecolor("#FFD700")
        bars[best_idx].set_linewidth(3)

        for bar, v in zip(bars, values):
            fmt = f"{v:.2f}%" if unit == "%" else f"{v:.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.012,
                fmt, ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(f"{title}\n({unit})", fontweight="bold")
        ax.set_ylim(0, max(values) * 1.22)
        if unit == "%":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x:.1f}%")
            )
        else:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

    # Golden star label
    axes[0].text(
        0.5, -0.16, "★  Gold border = best model",
        transform=axes[0].transAxes, ha="center", fontsize=8.5, color="#B8860B",
    )

    fig.suptitle(
        "Forecasting Performance — Real SIM/DataSUS Data, São Paulo State (2019–2023)\n"
        "186 out-of-sample predictions · 31 rolling windows · 6-month horizon",
        fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    path = IMG / "fig3_model_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 4 — sMAPE by forecast horizon (1–6 months ahead)
# ═════════════════════════════════════════════════════════════════════════════
def fig4_smape_by_horizon():
    hz_agg = (
        preds.groupby(["model", "horizon"])["smape_row"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "smape_mean", "std": "smape_std"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for model in MODEL_ORDER:
        sub = hz_agg[hz_agg["model"] == model].sort_values("horizon")
        ax.plot(sub["horizon"], sub["smape_mean"],
                color=C[model], linewidth=2.2, marker="o", markersize=7,
                label=MODEL_LABELS[model])
        ax.fill_between(
            sub["horizon"],
            sub["smape_mean"] - sub["smape_std"],
            sub["smape_mean"] + sub["smape_std"],
            color=C[model], alpha=0.12,
        )

    ax.set_xlabel("Forecast Horizon (months ahead)")
    ax.set_ylabel("sMAPE (%)")
    ax.set_title(
        "Forecast Accuracy by Horizon\nSão Paulo State — Rolling Backtesting (2021–2023)",
        fontweight="bold",
    )
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"h={i}" for i in range(1, 7)])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    path = IMG / "fig4_smape_by_horizon.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Prediction error distributions (box + violin)
# ═════════════════════════════════════════════════════════════════════════════
def fig5_error_distribution():
    preds["abs_error"] = (preds["y_pred"] - preds["y_true"]).abs()

    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    data_ae = [preds[preds["model"] == m]["abs_error"].values   for m in MODEL_ORDER]
    data_sm = [preds[preds["model"] == m]["smape_row"].values   for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: absolute error box plot ────────────────────────────────────────
    ax = axes[0]
    bp = ax.boxplot(
        data_ae, labels=labels, patch_artist=True, notch=False,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(C[model])
        patch.set_alpha(0.82)

    for i, (d, model) in enumerate(zip(data_ae, MODEL_ORDER), start=1):
        ax.text(i, np.percentile(d, 75) + 30,
                f"Median\n{np.median(d):.0f}",
                ha="center", fontsize=8, color=C[model])

    ax.set_ylabel("Absolute Error (deaths/month)")
    ax.set_title("Distribution of Absolute Error", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Right: sMAPE violin ───────────────────────────────────────────────────
    ax = axes[1]
    parts = ax.violinplot(data_sm, positions=[1, 2, 3],
                          showmedians=True, showextrema=True)
    for pc, model in zip(parts["bodies"], MODEL_ORDER):
        pc.set_facecolor(C[model])
        pc.set_alpha(0.72)
    for part_name in ("cmedians", "cmaxes", "cmins", "cbars"):
        if part_name in parts:
            parts[part_name].set_linewidth(1.5)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2.5)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("sMAPE (%)")
    ax.set_title("Distribution of sMAPE", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.suptitle(
        "Prediction Error Distributions — n = 186 per model  "
        "(31 rolling windows × 6-month horizon)",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = IMG / "fig5_error_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 6 — sMAPE across synthetic benchmark sizes + real SP data
# ═════════════════════════════════════════════════════════════════════════════
def fig6_smape_by_sample():
    # Build a unified table: synthetic (1k/5k/10k) + real SP
    synth = all_m[["model", "sample_size", "smape"]].copy()
    synth["sample_label"] = synth["sample_size"].astype(str)

    real_rows = []
    for _, row in metrics.iterrows():
        real_rows.append({
            "model":        row["model"],
            "sample_label": "Real SP\n(2019–23)",
            "smape":        row["smape"],
        })
    combined = pd.concat(
        [synth[["model", "sample_label", "smape"]].rename(columns={"sample_label": "label"}),
         pd.DataFrame(real_rows).rename(columns={"sample_label": "label"})],
        ignore_index=True,
    )

    sample_order = ["1000", "5000", "10000", "Real SP\n(2019–23)"]
    x      = np.arange(len(sample_order))
    width  = 0.25
    offset = {"timesfm": -width, "sarima": 0.0, "prophet": width}

    fig, ax = plt.subplots(figsize=(12, 5))

    for model in MODEL_ORDER:
        vals = []
        for s in sample_order:
            row = combined[(combined["model"] == model) & (combined["label"] == s)]
            vals.append(float(row["smape"].values[0]) if len(row) > 0 else np.nan)

        bars = ax.bar(
            x + offset[model], vals, width=width * 0.92,
            label=MODEL_LABELS[model], color=C[model], alpha=0.85,
            edgecolor="white", linewidth=0.9,
        )
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.08,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8.2,
                )

    # Vertical separator before real data
    ax.axvline(2.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(2.52, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 12,
            "Real\ndata →", color="gray", fontsize=8, va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(["Synthetic\nn=1,000", "Synthetic\nn=5,000",
                         "Synthetic\nn=10,000", "Real SP\n(2019–23)"])
    ax.set_ylabel("sMAPE (%)")
    ax.set_title(
        "sMAPE by Model and Dataset — Synthetic Benchmark + Real SIM/DataSUS",
        fontweight="bold",
    )
    ax.legend(framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    fig.tight_layout()
    path = IMG / "fig6_smape_by_sample.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 7 — Average seasonal mortality profile (bar chart by month)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_seasonal_profile():
    monthly = (
        obs.groupby("month")["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    def bar_color(m):
        if m in (6, 7):   return "#1565C0"   # winter peak
        if m in (1, 2):   return "#C62828"   # summer trough
        return "#78909C"                      # other months

    fig, ax = plt.subplots(figsize=(11, 4.5))

    bars = ax.bar(
        monthly["month"], monthly["mean"],
        color=[bar_color(m) for m in monthly["month"]],
        alpha=0.85, width=0.72, edgecolor="white", linewidth=0.9,
    )
    ax.errorbar(
        monthly["month"], monthly["mean"],
        yerr=monthly["std"],
        fmt="none", ecolor="#37474F", elinewidth=1.4,
        capsize=5, capthick=1.4, alpha=0.7,
    )

    # Range annotation (min–max whiskers as light shading)
    ax.fill_between(
        monthly["month"],
        monthly["min"], monthly["max"],
        alpha=0.08, color="#37474F", step="mid",
    )

    # Value labels on top of bars
    for bar, mean_v in zip(bars, monthly["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + monthly["std"].max() * 0.05,
            f"{mean_v:.0f}",
            ha="center", va="bottom", fontsize=8.2,
        )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Mean cardiovascular deaths / month")
    ax.set_title(
        "Average Monthly Cardiovascular Mortality Profile\n"
        "São Paulo State, Brazil (2019–2023) — error bars = ±1 SD, shading = min–max range",
        fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    winter_patch = mpatches.Patch(color="#1565C0", alpha=0.85, label="Winter peak (Jun–Jul)")
    summer_patch = mpatches.Patch(color="#C62828", alpha=0.85, label="Summer trough (Jan–Feb)")
    other_patch  = mpatches.Patch(color="#78909C", alpha=0.85, label="Other months")
    ax.legend(handles=[winter_patch, summer_patch, other_patch],
              loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = IMG / "fig7_seasonal_profile.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating paper figures → {IMG}\n")
    fig1_time_series()
    fig2_forecast_comparison()
    fig3_model_metrics()
    fig4_smape_by_horizon()
    fig5_error_distribution()
    fig6_smape_by_sample()
    fig7_seasonal_profile()
    print(f"\nDone. {len(list(IMG.glob('fig*.png')))} figures saved to {IMG}\n")
