"""Incerteza do benchmark: IC por bootstrap de janelas, teste pareado e erro por horizonte.

Motivação: o benchmark publicado ordena TimesFM, SARIMA e Prophet por uma amplitude de
0,38 pp de sMAPE, sem nenhuma medida de incerteza. Antes de comparar mais famílias de modelo
é preciso saber a largura do intervalo de uma família, senão o ranking é ruído
(aprendizado 5 de `ai-lab-hub/docs/aprendizados-pipeline-agentes.md`).

Desenho da reamostragem: as previsões de cada modelo NÃO são independentes. Elas vêm de
janelas de rolling origin que se sobrepõem, e dentro de cada janela os horizontes
compartilham a mesma origem. Reamostrar previsão a previsão superestimaria a precisão. Por isso
o bootstrap reamostra **janelas inteiras**, com os horizontes juntos, e usa as MESMAS janelas
para todos os modelos em cada réplica, o que preserva o pareamento.

Requer um CSV de previsões com colunas: model, date, y_true, y_pred, window, horizon
(formato gravado por scripts/run_benchmark.py).

Uso:
    python scripts/run_uncertainty.py --predictions-csv results/benchmark_..._predictions.csv \
        --models timesfm,sarima,prophet --output-prefix results/uncertainty_2010_2023
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from cv_timeseries.uncertainty import (
    bootstrap_metrica,
    dm_test,
    ic_percentil,
    p_bicaudal,
    sortear_janelas,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incerteza do benchmark (bootstrap + DM)")
    parser.add_argument(
        "--predictions-csv",
        default=str(ROOT / "results" / "predictions_indexed.csv"),
        help="CSV de previsões com colunas model,date,y_true,y_pred,window,horizon",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Modelos a comparar, separados por vírgula (default: todos no CSV)",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(ROOT / "results" / "uncertainty"),
        help="Prefixo dos CSVs de saída (_bootstrap_metrics, _pairwise_tests, _error_by_horizon)",
    )
    parser.add_argument("--n-boot", type=int, default=10_000, help="Réplicas de bootstrap")
    parser.add_argument("--seed", type=int, default=20260801, help="Semente do gerador")
    return parser.parse_args()


def smape_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """sMAPE ponto a ponto, em porcentagem."""
    return 200.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))


def to_matrices(df: pd.DataFrame, models: list[str], n_win: int, n_hor: int) -> dict:
    """Empilha as previsões em matriz (janela x horizonte) por modelo."""
    out = {}
    for m in models:
        g = df[df.model == m].sort_values(["window", "horizon"])
        if len(g) != n_win * n_hor:
            raise ValueError(
                f"Modelo {m} tem {len(g)} previsões; esperado {n_win}x{n_hor}. "
                "O bootstrap pareado exige o mesmo conjunto de janelas para todos os modelos."
            )
        out[m] = {
            "y_true": g["y_true"].to_numpy().reshape(n_win, n_hor),
            "y_pred": g["y_pred"].to_numpy().reshape(n_win, n_hor),
        }
    return out


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.predictions_csv, parse_dates=["date"])
    for col in ("model", "y_true", "y_pred", "window", "horizon"):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente em {args.predictions_csv}: {col}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        models = sorted(df.model.unique())
    missing = set(models) - set(df.model.unique())
    if missing:
        raise ValueError(f"Modelos ausentes no CSV: {sorted(missing)}")

    n_win = int(df.window.max())
    n_hor = int(df.horizon.max())
    mats = to_matrices(df, models, n_win, n_hor)
    B = args.n_boot

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_boot = out_prefix.with_name(out_prefix.name + "_bootstrap_metrics.csv")
    out_pair = out_prefix.with_name(out_prefix.name + "_pairwise_tests.csv")
    out_horiz = out_prefix.with_name(out_prefix.name + "_error_by_horizon.csv")

    # sMAPE e erro absoluto ponto a ponto, matriz (janela x horizonte) por modelo
    sm = {m: smape_vec(mats[m]["y_true"], mats[m]["y_pred"]) for m in models}
    ae = {m: np.abs(mats[m]["y_pred"] - mats[m]["y_true"]) for m in models}

    # ---------- N2 e N3: bootstrap pareado de janelas ----------
    idx = sortear_janelas(n_win, B, args.seed)
    boot_sm = {m: bootstrap_metrica(sm[m], idx) for m in models}
    boot_ae = {m: bootstrap_metrica(ae[m], idx) for m in models}

    linhas = []
    for m in models:
        for nome, pontual, boot in (
            ("smape", sm[m].mean(), boot_sm[m]),
            ("mae", ae[m].mean(), boot_ae[m]),
        ):
            lo, hi = ic_percentil(boot)
            linhas.append({
                "model": m, "metric": nome, "point": pontual,
                "ci_low": lo, "ci_high": hi, "ci_width": hi - lo,
                "n_windows": n_win, "n_predictions": sm[m].size, "B": B,
            })
    boot_df = pd.DataFrame(linhas)
    boot_df.to_csv(out_boot, index=False)

    pares = list(combinations(models, 2))
    linhas = []
    for a, b in pares:
        for nome, mat, boot in (
            ("smape", sm, boot_sm),
            ("mae", ae, boot_ae),
        ):
            dif = mat[a].mean() - mat[b].mean()
            boot_dif = boot[a] - boot[b]
            lo, hi = ic_percentil(boot_dif)
            p_boot = p_bicaudal(boot_dif)
            linhas.append({
                "pair": f"{a}_menos_{b}", "metric": nome, "diff_point": dif,
                "ci_low": lo, "ci_high": hi, "ci_contains_zero": bool(lo <= 0 <= hi),
                "p_bootstrap": p_boot, "B": B,
            })
    pair_df = pd.DataFrame(linhas)

    # Diebold-Mariano por horizonte, perda = erro absoluto
    dm_linhas = []
    for a, b in pares:
        for h in range(1, n_hor + 1):
            d = ae[a][:, h - 1] - ae[b][:, h - 1]
            estat, p = dm_test(d, h)
            dm_linhas.append({
                "pair": f"{a}_menos_{b}", "horizon": h, "mean_loss_diff": float(d.mean()),
                "dm_stat_harvey": estat, "p_value": p, "n_windows": n_win,
            })
    dm_df = pd.DataFrame(dm_linhas)
    pd.concat([
        pair_df.assign(test="bootstrap_pareado", horizon="pooled"),
        dm_df.assign(test="diebold_mariano_harvey", metric="mae"),
    ], ignore_index=True).to_csv(out_pair, index=False)

    # ---------- N4: erro por horizonte ----------
    linhas = []
    for m in models:
        for h in range(1, n_hor + 1):
            col = sm[m][:, h - 1]
            # Reusa o MESMO sorteio de janelas do bloco pooled, em vez de sortear
            # dentro do loop. Sortear aqui tornava o resultado dependente da ordem
            # em que os modelos são passados, porque cada iteração consumia estado
            # do gerador. Reusar `idx` também mantém as janelas pareadas entre
            # modelos e horizontes, que é o mesmo desenho dos blocos N2 e N3.
            boot_h = col[idx].mean(axis=1)
            lo, hi = ic_percentil(boot_h)
            linhas.append({
                "model": m, "horizon": h, "smape": col.mean(),
                "ci_low": lo, "ci_high": hi,
                "mae": ae[m][:, h - 1].mean(), "n_windows": n_win,
            })
    hor_df = pd.DataFrame(linhas)
    hor_df.to_csv(out_horiz, index=False)

    # ---------- relatório ----------
    print(f"N2  IC95% por bootstrap de janelas (B={B:,}, reamostra as {n_win} janelas inteiras)\n")
    print(f"{'modelo':<14}{'sMAPE':>8}{'IC95%':>22}{'largura':>10}")
    for _, r in boot_df[boot_df.metric == "smape"].iterrows():
        ic = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        print(f"{r.model:<14}{r.point:>8.2f}{ic:>22}{r.ci_width:>10.2f}")
    amplitude = boot_df[boot_df.metric == "smape"].point.max() - boot_df[boot_df.metric == "smape"].point.min()
    largura_media = boot_df[boot_df.metric == "smape"].ci_width.mean()
    print(f"\nAmplitude entre modelos: {amplitude:.2f} pp")
    print(f"Largura média do IC:     {largura_media:.2f} pp")
    print(f"Razão amplitude/largura: {amplitude / largura_media:.2f}"
          f"  ({'ranking informativo' if amplitude > largura_media else 'ranking dentro do ruído'})")

    print("\n\nN3  Diferença pareada (bootstrap sobre as mesmas janelas)\n")
    print(f"{'par':<32}{'métrica':<8}{'diferença':>11}{'IC95%':>24}  zero dentro?")
    for _, r in pair_df.iterrows():
        ic = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
        print(f"{r.pair:<32}{r.metric:<8}{r.diff_point:>11.3f}{ic:>24}  "
              f"{'SIM' if r.ci_contains_zero else 'não'}  p={r.p_bootstrap:.3f}")

    print("\nDiebold-Mariano por horizonte (perda = erro absoluto, correção de Harvey)\n")
    print(f"{'par':<32}" + "".join(f"{'h=' + str(h):>12}" for h in range(1, n_hor + 1)))
    for a, b in pares:
        sub = dm_df[dm_df.pair == f"{a}_menos_{b}"].sort_values("horizon")
        cel = "".join(f"{r.p_value:>12.3f}" for _, r in sub.iterrows())
        print(f"{a + '_menos_' + b:<32}{cel}")
    print("(valores são p-valores; nenhum abaixo de 0,05 significa nenhuma diferença detectada)")

    print(f"\n\nN4  sMAPE por horizonte\n")
    print(f"{'modelo':<14}" + "".join(f"{'h=' + str(h):>9}" for h in range(1, n_hor + 1)))
    for m in models:
        sub = hor_df[hor_df.model == m].sort_values("horizon")
        print(f"{m:<14}" + "".join(f"{r.smape:>9.2f}" for _, r in sub.iterrows()))
    melhor = hor_df.loc[hor_df.groupby("horizon").smape.idxmin()].sort_values("horizon")
    print("\nMelhor modelo por horizonte: "
          + ", ".join(f"h{int(r.horizon)}={r.model}" for _, r in melhor.iterrows()))

    print(f"\nGravado: {out_boot}")
    print(f"Gravado: {out_pair}")
    print(f"Gravado: {out_horiz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
