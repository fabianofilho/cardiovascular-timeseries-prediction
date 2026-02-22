#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa uma amostra pequena via PySUS e cria série mensal date,value"
    )
    parser.add_argument("--source", choices=["sih", "sim"], default="sih")
    parser.add_argument("--uf", default="SP", help="UF (ex: SP)")
    parser.add_argument("--year", type=int, default=2024, help="Ano de referência")
    parser.add_argument(
        "--month",
        type=int,
        default=1,
        help="Mês (apenas para SIH, quando aplicável)",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Limite de linhas para teste rápido",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/pysus_sample_raw.csv",
        help="CSV de saída com amostra bruta",
    )
    parser.add_argument(
        "--series-output",
        default="data/processed/serie_eventos_sp_sample.csv",
        help="CSV de saída da série mensal (date,value)",
    )
    return parser.parse_args()


def _to_dataframe(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, tuple)):
        parts = [x for x in obj if isinstance(x, pd.DataFrame)]
        if parts:
            return pd.concat(parts, ignore_index=True)
    raise TypeError(f"Download retornou tipo não suportado: {type(obj)}")


def download_sample(source: str, uf: str, year: int, month: int) -> pd.DataFrame:
    if source == "sih":
        from pysus.online_data.SIH import download

        raw = download(uf, year, month)
        return _to_dataframe(raw)

    from pysus.online_data.SIM import download

    raw = download(uf, year)
    return _to_dataframe(raw)


def infer_date_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "DT_INTER",
        "DT_INTERNA",
        "DT_SAIDA",
        "DTOBITO",
        "DT_OBITO",
        "data",
        "date",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_monthly_series(df: pd.DataFrame, default_year: int, default_month: int) -> pd.DataFrame:
    date_col = infer_date_column(df)

    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
    elif {"ANO_CMPT", "MES_CMPT"}.issubset(df.columns):
        y = pd.to_numeric(df["ANO_CMPT"], errors="coerce")
        m = pd.to_numeric(df["MES_CMPT"], errors="coerce")
        dates = pd.to_datetime(
            pd.DataFrame({"year": y, "month": m, "day": 1}),
            errors="coerce",
        )
    else:
        dates = pd.to_datetime(f"{default_year:04d}-{default_month:02d}-01")
        dates = pd.Series([dates] * len(df))

    tmp = pd.DataFrame({"date": dates}).dropna()
    if tmp.empty:
        raise ValueError("Não foi possível inferir datas válidas na amostra baixada.")

    tmp["date"] = tmp["date"].dt.to_period("M").dt.to_timestamp("MS")
    series_df = (
        tmp.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "value"})
        .sort_values("date")
    )
    return series_df


def main() -> None:
    args = parse_args()

    df = download_sample(args.source, args.uf, args.year, args.month)
    if args.sample_rows > 0:
        df = df.head(args.sample_rows).copy()

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_output, index=False)

    series_df = build_monthly_series(df, default_year=args.year, default_month=args.month)

    series_output = Path(args.series_output)
    series_output.parent.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(series_output, index=False)

    print(f"[INFO] Amostra bruta salva em: {raw_output} (linhas={len(df)})")
    print(f"[INFO] Série mensal salva em: {series_output} (pontos={len(series_df)})")


if __name__ == "__main__":
    main()
