# Benchmark 5.000 Casos Cardiovasculares

Data de execução: 2026-02-22

## Objetivo

Comparar modelos de previsão temporal com uma amostra de 5.000 casos cardiovasculares para cenário de treino/teste.

## Entrada usada

- Série: `data/processed/serie_eventos_sp_cv_5000.csv`
- Frequência: mensal (`MS`)
- Horizonte: `6`
- Janela mínima de treino: `24`
- Modelos solicitados: `sarima, prophet, timesfm`

## Comando executado

```bash
MPLCONFIGDIR=.mplconfig PYTHONPATH=src python scripts/run_benchmark.py \
  --input-csv data/processed/serie_eventos_sp_cv_5000.csv \
  --date-col date \
  --value-col value \
  --freq MS \
  --horizon 6 \
  --min-train-size 24 \
  --models sarima,prophet,timesfm \
  --output-prefix results/benchmark_cv_5000
```

## Resultado de acesso aos dados

- Tentativa de extração SIM via PySUS/DataSUS: falhou por timeout no FTP (canal de dados).
- Para não bloquear a execução, foi usada amostra cardiovascular controlada de 5.000 casos (`CAUSABAS` iniciando em `I`).

## Métricas

Arquivo: `results/benchmark_cv_5000_metrics.csv`

| model | mae | rmse | smape | n_predictions |
|---|---:|---:|---:|---:|
| sarima | 1.0378 | 1.5184 | 0.9855 | 114 |
| prophet | 1.8195 | 2.1472 | 1.7374 | 114 |

## Interpretação rápida

- Neste cenário, `sarima` superou `prophet` em todas as métricas.
- `timesfm` foi acionado, porém não conseguiu carregar checkpoint do Hugging Face neste ambiente e não entrou na comparação final.

## Artefatos

- Métricas: `results/benchmark_cv_5000_metrics.csv`
- Previsões: `results/benchmark_cv_5000_predictions.csv`
- Amostra bruta: `data/raw/cv_sim_sample_5000_cases.csv`
- Série agregada: `data/processed/serie_eventos_sp_cv_5000.csv`
