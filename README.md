# Predição de Eventos Cardiovasculares em São Paulo

Repositório para modelagem de tendência de eventos cardiovasculares, com foco em:

- **Internações** (SIH/SUS)
- **Óbitos** (SIM)

## Objetivo

Comparar três famílias de modelos para previsão de séries temporais:

1. **ARIMA/SARIMA** (baseline estatístico robusto)
2. **Prophet** (sazonalidade e tendência com feriados/regressores)
3. **TimesFM** (foundation model para forecasting)

## Estrutura

- `data/raw/`: extrações brutas (SIH/SIM)
- `data/processed/`: séries tratadas (mensal)
- `src/cv_timeseries/`: código-fonte de preparação, modelos e avaliação
- `scripts/run_benchmark.py`: CLI para rodar benchmark
- `results/`: métricas e previsões

## Como começar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Formato mínimo do CSV de entrada

O script espera um CSV com, no mínimo:

- `date` (ou coluna equivalente informada no CLI)
- `value` (contagem de eventos no período)

Exemplo:

```csv
date,value
2015-01-01,120
2015-02-01,131
...
```

## Rodar benchmark

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --input-csv data/processed/serie_eventos_sp.csv \
  --date-col date \
  --value-col value \
  --freq MS \
  --horizon 6 \
  --output-prefix results/benchmark_sp
```

Saídas:

- `results/benchmark_sp_metrics.csv`
- `results/benchmark_sp_predictions.csv`

## Recomendação técnica inicial

Sem benchmark local, a melhor prática é:

- usar **SARIMA** como baseline obrigatório;
- usar **Prophet** quando houver sazonalidade forte e necessidade de interpretabilidade;
- testar **TimesFM** como candidato de maior desempenho, validando custo, estabilidade e disponibilidade de inferência.

A escolha final deve ser por **backtesting temporal** (rolling origin), não por preferência teórica.

## Próximos passos

1. Extrair SIH/SIM para série mensal por município/DRS.
2. Criar variáveis exógenas (temperatura, cobertura APS, envelhecimento, etc.).
3. Rodar benchmark por estrato (sexo, faixa etária, CID, território).
4. Definir modelo campeão por métrica (MAE/RMSE/sMAPE) e robustez operacional.
