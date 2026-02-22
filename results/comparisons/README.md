# Comparativo de Modelos por Tamanho de Amostra

## Melhor modelo por amostra (menor sMAPE)

| sample_size | model | mae | rmse | smape | n_predictions |
| --- | --- | --- | --- | --- | --- |
| 1000 | sarima | 0.9034360966366732 | 2.955657562377413 | 3.763137820625096 | 99 |
| 5000 | timesfm | 0.8199715865285773 | 1.077242279394319 | 0.760909621161365 | 114 |
| 10000 | sarima | 1.0680087369061453 | 1.2768263759377734 | 0.5005444214914281 | 114 |

## sMAPE por modelo e amostra

| model | 1000 | 5000 | 10000 |
| --- | --- | --- | --- |
| prophet | 12.273377277885729 | 1.7373582230860871 | 1.233644810925936 |
| sarima | 3.763137820625096 | 0.9854657611896572 | 0.5005444214914281 |
| timesfm | nan | 0.760909621161365 | 0.6801833613021016 |

## Arquivos base usados
- `results/benchmark_cv_1000_metrics.csv`
- `results/benchmark_cv_5000_metrics.csv`
- `results/benchmark_cv_10000_metrics.csv`