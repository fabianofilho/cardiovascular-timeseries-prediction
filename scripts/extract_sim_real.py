#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai SIM real via PySUS com retry e gera série mensal cardiovascular"
    )
    parser.add_argument("--uf", default="SP", help="UF (ex: SP)")
    parser.add_argument("--year", type=int, required=True, help="Ano do SIM")
    parser.add_argument("--cid-prefix", default="I", help="Prefixo CID cardiovascular")
    parser.add_argument("--max-rows", type=int, default=0, help="Limite opcional de linhas (0=sem limite)")
    parser.add_argument("--max-retries", type=int, default=3, help="Número máximo de tentativas")
    parser.add_argument("--retry-wait", type=int, default=20, help="Espera entre tentativas (s)")
    parser.add_argument("--raw-output", default="data/raw/sim_real_sp_latest.csv", help="CSV bruto de saída")
    parser.add_argument(
        "--series-output",
        default="data/processed/serie_eventos_sp_sim_real.csv",
        help="CSV date,value de saída",
    )
    parser.add_argument(
        "--meta-output",
        default="data/processed/sim_real_extraction_metadata.json",
        help="JSON com metadados da extração",
    )
    return parser.parse_args()


def _to_dataframe(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, tuple)):
        parts = [x for x in obj if isinstance(x, pd.DataFrame)]
        if parts:
            return pd.concat(parts, ignore_index=True)
    raise TypeError(f"Retorno do download em tipo não suportado: {type(obj)}")


def _infer_date_col(df: pd.DataFrame) -> str:
    candidates = ["DTOBITO", "DT_OBITO", "DT_OBITO_ORIG", "event_date", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Não foi possível identificar coluna de data de óbito no SIM.")


def extract_sim(uf: str, year: int, max_retries: int, retry_wait: int) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            from pysus.online_data.SIM import download

            print(f"[INFO] Tentativa {attempt}/{max_retries}: download SIM {uf}-{year}")
            obj = download(uf, year)
            df = _to_dataframe(obj)
            print(f"[INFO] Download concluído: {len(df)} linhas")
            return df
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Falha na tentativa {attempt}: {exc}")
            if attempt < max_retries:
                print(f"[INFO] Aguardando {retry_wait}s para nova tentativa...")
                time.sleep(retry_wait)

    raise RuntimeError(f"Extração SIM falhou após {max_retries} tentativas: {last_error}")


def build_outputs(df: pd.DataFrame, cid_prefix: str, max_rows: int):
    cid_col = "CAUSABAS" if "CAUSABAS" in df.columns else None
    if cid_col is None:
        raise ValueError("Coluna CAUSABAS não encontrada no dataset SIM baixado.")

    work = df.copy()
    work[cid_col] = work[cid_col].astype(str).str.upper().str.strip()
    work = work[work[cid_col].str.startswith(cid_prefix.upper())].copy()

    if max_rows > 0:
        work = work.head(max_rows).copy()

    if work.empty:
        raise ValueError("Sem registros após filtro cardiovascular.")

    date_col = _infer_date_col(work)
    work["event_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["event_date"]).copy()

    if work.empty:
        raise ValueError("Sem datas válidas após parse de data.")

    series_df = (
        work.assign(date=work["event_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "value"})
        .sort_values("date")
    )

    return work, series_df, cid_col, date_col


def main() -> None:
    args = parse_args()

    raw_path = Path(args.raw_output)
    series_path = Path(args.series_output)
    meta_path = Path(args.meta_output)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    series_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        df = extract_sim(
            uf=args.uf,
            year=args.year,
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        filtered_df, series_df, cid_col, date_col = build_outputs(
            df,
            cid_prefix=args.cid_prefix,
            max_rows=args.max_rows,
        )
        finished_at = datetime.now(timezone.utc).isoformat()

        filtered_df.to_csv(raw_path, index=False)
        series_df.to_csv(series_path, index=False)

        meta = {
            "source": "SIM",
            "uf": args.uf,
            "year": args.year,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "status": "success",
            "rows_downloaded": int(len(df)),
            "rows_after_cid_filter": int(len(filtered_df)),
            "series_points": int(len(series_df)),
            "cid_column": cid_col,
            "date_column": date_col,
            "cid_prefix": args.cid_prefix.upper(),
            "synthetic": False,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[INFO] Bruto salvo em: {raw_path}")
        print(f"[INFO] Série salva em: {series_path}")
        print(f"[INFO] Metadados salvos em: {meta_path}")
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        meta = {
            "source": "SIM",
            "uf": args.uf,
            "year": args.year,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "status": "failed",
            "error": str(exc),
            "cid_prefix": args.cid_prefix.upper(),
            "synthetic": False,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ERROR] Extração falhou. Metadados salvos em: {meta_path}")
        raise


if __name__ == "__main__":
    main()
