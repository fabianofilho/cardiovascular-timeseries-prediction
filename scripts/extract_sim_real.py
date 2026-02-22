#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from ftplib import FTP
from pathlib import Path

import pandas as pd
import requests


MIRROR_BASE = "https://datasus-ftp-mirror.nyc3.cdn.digitaloceanspaces.com"
FTP_HOST = "ftp.datasus.gov.br"
FTP_SIM_PATH = "/dissemin/publicos/SIM/CID10/DORES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai SIM real do DataSUS com estratégia de download em camadas"
    )
    parser.add_argument("--uf", default="SP", help="UF (ex: SP)")
    parser.add_argument(
        "--years",
        required=True,
        help="Anos: único (2022), lista (2019,2020,2022) ou range (2019-2023)",
    )
    parser.add_argument("--cid-prefix", default="I", help="Prefixo CID cardiovascular")
    parser.add_argument("--max-rows", type=int, default=0, help="Limite opcional de linhas (0=sem limite)")
    parser.add_argument("--max-retries", type=int, default=3, help="Tentativas por tier")
    parser.add_argument("--retry-wait", type=int, default=20, help="Espera entre tentativas (s)")
    parser.add_argument("--raw-output", default="data/raw/sim_real_sp_latest.csv")
    parser.add_argument("--series-output", default="data/processed/serie_eventos_sp_sim_real.csv")
    parser.add_argument("--meta-output", default="data/processed/sim_real_extraction_metadata.json")
    return parser.parse_args()


def parse_years(years_str: str) -> list[int]:
    if "-" in years_str and "," not in years_str:
        parts = years_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(y.strip()) for y in years_str.split(",")]


# ---------------------------------------------------------------------------
# Tier 1: HTTP mirror (CDN, mais confiável)
# ---------------------------------------------------------------------------

def download_via_http_mirror(uf: str, year: int, cache_dir: Path) -> Path:
    filename = f"DO{uf.upper()}{year}.dbc"
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"[INFO] Cache encontrado: {local_path}")
        return local_path

    url = f"{MIRROR_BASE}/SIM/CID10/DORES/{filename}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando via HTTP mirror: {url}")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            f.write(chunk)

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Download HTTP concluído: {local_path} ({size_mb:.1f} MB)")
    return local_path


# ---------------------------------------------------------------------------
# Tier 2: PySUS API corrigida
# ---------------------------------------------------------------------------

def download_via_pysus(uf: str, year: int) -> pd.DataFrame:
    from pysus.online_data.SIM import download

    print(f"[INFO] Baixando via PySUS: SIM CID10 {uf} {year}")
    result = download(groups="CID10", states=uf, years=year)
    return _result_to_dataframe(result)


def _result_to_dataframe(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, (list, tuple)):
        frames = []
        for item in result:
            if isinstance(item, pd.DataFrame):
                frames.append(item)
            elif isinstance(item, (str, Path)):
                frames.append(pd.read_parquet(str(item)))
            elif hasattr(item, "to_dataframe"):
                frames.append(item.to_dataframe())
        if frames:
            return pd.concat(frames, ignore_index=True)
    if isinstance(result, (str, Path)):
        return pd.read_parquet(str(result))
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    raise TypeError(f"Retorno PySUS não suportado: {type(result)}")


# ---------------------------------------------------------------------------
# Tier 3: FTP direto com timeout explícito
# ---------------------------------------------------------------------------

def download_via_ftp_direct(uf: str, year: int, cache_dir: Path, timeout: int = 60) -> Path:
    filename = f"DO{uf.upper()}{year}.dbc"
    local_path = cache_dir / filename

    if local_path.exists():
        print(f"[INFO] Cache encontrado: {local_path}")
        return local_path

    remote_path = f"{FTP_SIM_PATH}/{filename}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando via FTP direto: {FTP_HOST}{remote_path}")
    ftp = FTP(FTP_HOST, timeout=timeout)
    try:
        ftp.login()
        ftp.set_pasv(True)
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {remote_path}", f.write)
    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Download FTP concluído: {local_path} ({size_mb:.1f} MB)")
    return local_path


# ---------------------------------------------------------------------------
# DBC decoding
# ---------------------------------------------------------------------------

def decode_dbc(dbc_path: Path) -> pd.DataFrame:
    import tempfile

    from datasus_dbc import decompress_bytes
    from dbfread import DBF

    print(f"[INFO] Decodificando DBC: {dbc_path}")
    dbc_bytes = dbc_path.read_bytes()
    dbf_bytes = decompress_bytes(dbc_bytes)

    with tempfile.NamedTemporaryFile(suffix=".dbf", delete=False) as tmp:
        tmp.write(dbf_bytes)
        tmp_path = Path(tmp.name)

    try:
        dbf = DBF(str(tmp_path), encoding="latin1")
        df = pd.DataFrame(iter(dbf))
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"[INFO] DBC decodificado: {len(df)} linhas, {len(df.columns)} colunas")
    return df


# ---------------------------------------------------------------------------
# Extração com estratégia em camadas
# ---------------------------------------------------------------------------

def extract_sim_year(uf: str, year: int, max_retries: int, retry_wait: int) -> pd.DataFrame:
    cache_dir = Path("data/raw/cache")

    # Tier 1: HTTP mirror
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TIER1] HTTP mirror — tentativa {attempt}/{max_retries} para {uf}-{year}")
            dbc_path = download_via_http_mirror(uf, year, cache_dir)
            return decode_dbc(dbc_path)
        except Exception as exc:
            print(f"[WARN] Tier 1 falhou: {exc}")
            if attempt < max_retries:
                time.sleep(min(retry_wait // 2, 10))

    # Tier 2: PySUS API corrigida
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TIER2] PySUS API — tentativa {attempt}/{max_retries} para {uf}-{year}")
            return download_via_pysus(uf, year)
        except Exception as exc:
            print(f"[WARN] Tier 2 falhou: {exc}")
            if attempt < max_retries:
                time.sleep(retry_wait)

    # Tier 3: FTP direto com timeout
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TIER3] FTP direto — tentativa {attempt}/{max_retries} para {uf}-{year}")
            dbc_path = download_via_ftp_direct(uf, year, cache_dir, timeout=60)
            return decode_dbc(dbc_path)
        except Exception as exc:
            print(f"[WARN] Tier 3 falhou: {exc}")
            if attempt < max_retries:
                time.sleep(retry_wait)

    raise RuntimeError(f"Todos os tiers falharam para SIM {uf}-{year}")


def extract_sim(uf: str, years: list[int], max_retries: int, retry_wait: int) -> pd.DataFrame:
    frames = []
    for year in years:
        df = extract_sim_year(uf, year, max_retries, retry_wait)
        frames.append(df)
        print(f"[INFO] {uf}-{year}: {len(df)} linhas")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Processamento de saída
# ---------------------------------------------------------------------------

def _infer_date_col(df: pd.DataFrame) -> str:
    candidates = ["DTOBITO", "DT_OBITO", "DT_OBITO_ORIG", "event_date", "date"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Não foi possível identificar coluna de data de óbito no SIM.")


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
    raw_dates = work[date_col].astype(str).str.strip()
    # SIM uses ddmmyyyy format (e.g. 21042022)
    work["event_date"] = pd.to_datetime(raw_dates, format="%d%m%Y", errors="coerce")
    # Fallback for other formats
    mask_nat = work["event_date"].isna()
    if mask_nat.any():
        work.loc[mask_nat, "event_date"] = pd.to_datetime(
            raw_dates[mask_nat], format="mixed", dayfirst=True, errors="coerce"
        )
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
    years = parse_years(args.years)

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
            years=years,
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
            "years": years,
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
            "years": years,
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
