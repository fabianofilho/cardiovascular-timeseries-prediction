# Benchmark 10.000 Casos Cardiovasculares

Data de execução: 2026-02-22

## Configuração

- Série: `data/processed/serie_eventos_sp_cv_10000.csv`
- Frequência: mensal (`MS`)
- Horizonte: `6`
- Janela mínima de treino: `24`
- Modelos: `sarima, prophet, timesfm`
- TimesFM: checkpoint real via Hugging Face (`google/timesfm-2.5-200m-pytorch`)

## Métricas

Arquivo: `results/benchmark_cv_10000_metrics.csv`

| model | mae | rmse | smape | n_predictions |
|---|---:|---:|---:|---:|
| sarima | 1.0680 | 1.2768 | 0.5005 | 114 |
| timesfm | 1.4645 | 1.8987 | 0.6802 | 114 |
| prophet | 2.6241 | 3.5762 | 1.2336 | 114 |

## Conclusão rápida

- Nesta execução de 10k, `sarima` foi o melhor modelo por sMAPE.
- Comparativos entre 1k, 5k e 10k estão em `results/comparisons/`.
