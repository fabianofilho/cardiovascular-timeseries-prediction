#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida se o dataset é plausivelmente real e apto para benchmark"
    )
    parser.add_argument("--raw-csv", required=True, help="CSV bruto")
    parser.add_argument("--series-csv", required=True, help="CSV final date,value")
    parser.add_argument("--meta-json", default="", help="Metadado de extração")
    parser.add_argument("--min-series-points", type=int, default=24)
    parser.add_argument("--expected-cid-prefix", default="I")
    parser.add_argument("--report-output", default="results/data_reality_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = pd.read_csv(args.raw_csv)
    series = pd.read_csv(args.series_csv)

    checks = {}

    cid_col = "CAUSABAS" if "CAUSABAS" in raw.columns else ("causabas" if "causabas" in raw.columns else None)
    checks["has_cid_column"] = cid_col is not None
    checks["raw_rows_gt_0"] = len(raw) > 0

    if cid_col is not None:
        s = raw[cid_col].astype(str).str.upper().str.strip()
        checks["cid_prefix_all_match"] = bool(s.str.startswith(args.expected_cid_prefix.upper()).all())
        checks["cid_unique_ge_3"] = int(s.nunique()) >= 3
    else:
        checks["cid_prefix_all_match"] = False
        checks["cid_unique_ge_3"] = False

    checks["series_has_columns"] = set(["date", "value"]).issubset(series.columns)
    checks["series_points_min"] = len(series) >= args.min_series_points
    checks["series_value_sum_gt_0"] = float(series["value"].sum()) > 0 if "value" in series.columns else False

    if args.meta_json:
        meta_path = Path(args.meta_json)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            checks["meta_not_synthetic"] = not bool(meta.get("synthetic", True))
            checks["meta_has_source_sim"] = meta.get("source") == "SIM"
        else:
            checks["meta_not_synthetic"] = False
            checks["meta_has_source_sim"] = False

    passed = all(bool(v) for v in checks.values())

    report = {
        "passed": passed,
        "checks": checks,
        "raw_csv": args.raw_csv,
        "series_csv": args.series_csv,
    }

    out = Path(args.report_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
