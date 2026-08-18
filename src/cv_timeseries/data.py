from __future__ import annotations

import pandas as pd


def load_and_aggregate_series(
    csv_path: str,
    date_col: str,
    value_col: str,
    freq: str = "MS",
) -> pd.Series:
    """Carrega CSV e agrega para frequência temporal informada."""
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(f"Coluna de data não encontrada: {date_col}")
    if value_col not in df.columns:
        raise ValueError(f"Coluna de valor não encontrada: {value_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Existem datas inválidas no arquivo de entrada.")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if df[value_col].isna().any():
        raise ValueError("Existem valores não numéricos no arquivo de entrada.")

    series = (
        df.set_index(date_col)[value_col]
        .sort_index()
        .resample(freq)
        .sum(min_count=1)
        .fillna(0.0)
    )
    series.name = value_col
    return series


def checa_serie_agregada(series: pd.Series, freq: str = "MS") -> dict:
    """Checagens de sanidade sobre a serie ja agregada.

    Existe por causa de uma armadilha de `load_and_aggregate_series`: um periodo sem
    NENHUMA linha na entrada sai como 0, nao como faltante, porque o `.sum(min_count=1)`
    produz NaN e o `.fillna(0.0)` o converte. Para contagem de evento isso e defensavel;
    para mortalidade nao e. O minimo mensal real da serie de Sao Paulo e 5.811 obitos,
    entao um mes zerado significa falha de extracao, e entraria no benchmark como
    outlier extremo que os modelos tentariam ajustar.

    O que o validador conferia antes (numero de pontos e soma maior que zero) passa
    tranquilamente com um mes zerado no meio. Estas duas checagens fecham isso.

    Devolve dict com o resultado booleano de cada checagem e os periodos culpados,
    porque "reprovou" sem dizer onde nao ajuda ninguem a consertar.
    """
    zerados = series.index[series.values == 0]
    esperado = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    faltantes = esperado.difference(series.index)
    return {
        "series_no_zero_period": len(zerados) == 0,
        "series_no_date_gap": len(faltantes) == 0,
        "_detalhe": {
            "periodos_zerados": [str(d.date()) for d in zerados],
            "periodos_ausentes": [str(d.date()) for d in faltantes],
            "n_periodos": int(len(series)),
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
        },
    }
