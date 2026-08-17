#!/usr/bin/env python3
"""Baixa temperatura horária do INMET (BDMEP, estações automáticas) e agrega para mensal.

Fonte: zips anuais públicos em https://portal.inmet.gov.br/uploads/dadoshistoricos/{ano}.zip
Estação primária A701 (São Paulo - Mirante de Santana); fallbacks A755 (Barueri) e
A771/A711 conforme disponibilidade, usados apenas para preencher meses faltantes da primária.

Saída: CSV mensal com colunas date,tmin,tmed
    tmin = média mensal das mínimas diárias (driver fisiológico da sazonalidade CV)
    tmed = média mensal da temperatura de bulbo seco horária

Uso:
    python scripts/fetch_temperature_inmet.py --years 2010-2023 \
        --output results/series/temperatura_sp_mensal_2010_2023.csv
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"
CACHE_DIR = Path("data/raw/cache_inmet")

COL_TMIN = "TEMPERATURA M"  # "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)" — prefixo robusto a encoding
COL_TBULBO = "TEMPERATURA DO AR - BULBO SECO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temperatura mensal INMET para exógena")
    parser.add_argument("--years", default="2010-2023", help="Intervalo: 2010-2023")
    parser.add_argument("--station", default="A701", help="Código da estação primária")
    parser.add_argument(
        "--fallback-stations",
        default="A755,A711",
        help="Estações usadas somente para meses faltantes da primária",
    )
    parser.add_argument(
        "--output",
        default="results/series/temperatura_sp_mensal_2010_2023.csv",
    )
    parser.add_argument("--max-missing-interp", type=int, default=2,
                        help="Máximo de meses isolados a interpolar; acima disso, aborta")
    return parser.parse_args()


def year_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in spec.split(",")]


def download_zip(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{year}.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[INFO] {year}: cache em {dest}")
        return dest
    url = BASE_URL.format(year=year)
    print(f"[INFO] {year}: baixando {url}")
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def read_station_csv(zip_path: Path, station: str) -> pd.DataFrame | None:
    """Extrai e parseia o CSV horário de uma estação dentro do zip anual do INMET."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if f"_{station}_" in n.upper()]
        if not names:
            return None
        raw = zf.read(names[0])

    # Formato BDMEP: 8 linhas de metadados, ';' como separador, vírgula decimal, latin-1
    text = raw.decode("latin-1")
    df = pd.read_csv(
        io.StringIO(text),
        sep=";",
        skiprows=8,
        decimal=",",
        na_values=["-9999", ""],
    )
    date_col = next(c for c in df.columns if c.upper().startswith("DATA"))
    tmin_col = next((c for c in df.columns if c.upper().startswith(COL_TMIN.upper())
                     and "NIMA" in c.upper()), None)
    tbulbo_col = next((c for c in df.columns if c.upper().startswith(COL_TBULBO.upper())), None)
    if tmin_col is None or tbulbo_col is None:
        raise ValueError(f"Colunas de temperatura não encontradas em {zip_path.name}/{station}")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], format="mixed", dayfirst=False),
        "tmin_h": pd.to_numeric(df[tmin_col], errors="coerce"),
        "tbulbo_h": pd.to_numeric(df[tbulbo_col], errors="coerce"),
    })
    return out


def monthly_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mínima diária → média mensal (tmin); bulbo seco horário → média mensal (tmed)."""
    daily = hourly.groupby(hourly["date"].dt.date).agg(
        tmin_d=("tmin_h", "min"),
        tmed_d=("tbulbo_h", "mean"),
        n_horas=("tbulbo_h", "count"),
    )
    daily.index = pd.to_datetime(daily.index)
    # Dia com menos de 18 horas válidas não é confiável para a mínima
    daily = daily[daily["n_horas"] >= 18]
    monthly = daily.resample("MS").agg(
        tmin=("tmin_d", "mean"),
        tmed=("tmed_d", "mean"),
        n_dias=("tmin_d", "count"),
    )
    # Mês com menos de 20 dias válidos vira faltante
    monthly.loc[monthly["n_dias"] < 20, ["tmin", "tmed"]] = pd.NA
    return monthly[["tmin", "tmed"]]


def main() -> int:
    args = parse_args()
    years = year_range(args.years)
    stations = [args.station] + [s.strip() for s in args.fallback_stations.split(",") if s.strip()]

    per_station: dict[str, list[pd.DataFrame]] = {s: [] for s in stations}
    for year in years:
        zip_path = download_zip(year)
        for s in stations:
            hourly = read_station_csv(zip_path, s)
            if hourly is not None:
                per_station[s].append(hourly)
            elif s == args.station:
                print(f"[WARN] {year}: estação primária {s} ausente no zip")

    monthly_by_station = {}
    for s, frames in per_station.items():
        if frames:
            monthly_by_station[s] = monthly_from_hourly(pd.concat(frames, ignore_index=True))

    if args.station not in monthly_by_station:
        raise SystemExit("[ERRO] Estação primária sem nenhum dado.")

    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")
    result = monthly_by_station[args.station].reindex(idx)

    # Preenche faltantes da primária com fallbacks, na ordem
    for s in stations[1:]:
        if s not in monthly_by_station:
            continue
        fb = monthly_by_station[s].reindex(idx)
        missing = result["tmin"].isna()
        n_fill = int((missing & fb["tmin"].notna()).sum())
        if n_fill:
            print(f"[INFO] Preenchendo {n_fill} meses com {s}")
            result.loc[missing, ["tmin", "tmed"]] = fb.loc[missing, ["tmin", "tmed"]]

    still_missing = result["tmin"].isna()
    n_missing = int(still_missing.sum())
    if n_missing > args.max_missing_interp:
        print(result[still_missing])
        raise SystemExit(
            f"[ERRO] {n_missing} meses sem dado após fallbacks (limite "
            f"{args.max_missing_interp}). Use ERA5 como plano B."
        )
    if n_missing:
        print(f"[WARN] Interpolando {n_missing} meses isolados: "
              f"{[d.strftime('%Y-%m') for d in result[still_missing].index]}")
        result = result.interpolate(method="linear", limit=2)

    out = result.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(out_path, index=False)
    print(f"[INFO] {len(out)} meses gravados em {out_path}")
    print(out.describe().round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
