#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print('[CMD]', ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pipeline real: extrair, validar e benchmark')
    parser.add_argument('--uf', default='SP')
    parser.add_argument('--years', required=True, help='Anos: 2022, 2019-2023, ou 2019,2020,2022')
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retry-wait', type=int, default=20)
    parser.add_argument('--raw-output', default='data/raw/sim_real_sp_latest.csv')
    parser.add_argument('--series-output', default='data/processed/serie_eventos_sp_sim_real.csv')
    parser.add_argument('--meta-output', default='data/processed/sim_real_extraction_metadata.json')
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--min-train-size', type=int, default=24)
    parser.add_argument('--models', default='sarima,prophet,timesfm')
    parser.add_argument('--output-prefix', default='results/benchmark_sim_real_sp')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parents[1]

    run([
        py,
        str(root / 'scripts' / 'extract_sim_real.py'),
        '--uf', args.uf,
        '--years', args.years,
        '--max-retries', str(args.max_retries),
        '--retry-wait', str(args.retry_wait),
        '--raw-output', args.raw_output,
        '--series-output', args.series_output,
        '--meta-output', args.meta_output,
    ])

    run([
        py,
        str(root / 'scripts' / 'validate_real_dataset.py'),
        '--raw-csv', args.raw_output,
        '--series-csv', args.series_output,
        '--meta-json', args.meta_output,
        '--report-output', 'results/data_reality_report.json',
    ])

    run([
        py,
        str(root / 'scripts' / 'run_benchmark.py'),
        '--input-csv', args.series_output,
        '--date-col', 'date',
        '--value-col', 'value',
        '--freq', 'MS',
        '--horizon', str(args.horizon),
        '--min-train-size', str(args.min_train_size),
        '--models', args.models,
        '--output-prefix', args.output_prefix,
    ])


if __name__ == '__main__':
    main()
