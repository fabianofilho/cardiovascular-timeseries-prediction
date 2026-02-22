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
