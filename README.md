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
- `docs/`: dicionários CID e guias de colunas SIH/SIM
- `src/cv_timeseries/`: código-fonte de preparação, modelos e avaliação
- `scripts/run_benchmark.py`: CLI para rodar benchmark
- `results/`: métricas e previsões

## Dicionários e Regras de Incidência

- Dicionário CID-10 cardiovascular (SIM): `docs/cid10_cardiovascular_sim.md`
- Tabela de faixas CID para filtros automatizados: `docs/cid10_cardiovascular_ranges.csv`
- Colunas e regras SIH/SIM para incidência cardiovascular: `docs/sih_sim_colunas_incidencia.md`

## Como começar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências opcionais (modelos mais pesados):

```bash
pip install -r requirements-optional.txt
```

Atalhos com `make`:

```bash
make setup-base
make setup-full
```

## Fluxo recomendado: amostra pequena primeiro (PySUS)

### 1) Baixar amostra pequena SIH/SIM via PySUS

Exemplo SIH/SP (amostra curta para smoke test):

```bash
PYTHONPATH=src python scripts/prepare_pysus_sample.py \
  --source sih \
  --uf SP \
  --year 2024 \
  --month 1 \
  --sample-rows 5000 \
  --raw-output data/raw/pysus_sih_sp_2024_01_sample.csv \
  --series-output data/processed/serie_eventos_sp_sample.csv
```

O script gera:

- CSV bruto da amostra (`data/raw/...`)
- Série mensal agregada em formato `date,value` (`data/processed/...`)

Também disponível via atalho:

```bash
make sample-pysus
```

### 2) Rodar benchmark rápido (somente baseline)

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --input-csv data/processed/serie_eventos_sp_sample.csv \
  --date-col date \
  --value-col value \
  --freq MS \
  --horizon 3 \
  --min-train-size 12 \
  --models sarima \
  --output-prefix results/benchmark_sp_sample
```

Ou via atalho (série tiny local para smoke test):

```bash
make smoke-baseline
```

### 3) Rodar benchmark completo (quando ambiente estiver pronto)

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --input-csv data/processed/serie_eventos_sp_sample.csv \
  --models sarima,prophet,timesfm \
  --horizon 6 \
  --min-train-size 24 \
  --output-prefix results/benchmark_sp_sample_full
```

Ou via atalho:

```bash
make benchmark-all
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
  --models sarima,prophet,timesfm \
  --output-prefix results/benchmark_sp
```

Saídas:

- `results/benchmark_sp_metrics.csv`
- `results/benchmark_sp_predictions.csv`

## Resultados Atualizados (1k, 5k, 10k)

### Métricas por amostra (sMAPE menor = melhor)

| Amostra | Melhor modelo | sMAPE |
|---|---|---:|
| 1.000 | SARIMA | 3.7631 |
| 5.000 | TimesFM | 0.7609 |
| 10.000 | SARIMA | 0.5005 |

### Último benchmark (10k)

Arquivo: `results/benchmark_cv_10000_metrics.csv`

| model | mae | rmse | smape | n_predictions |
|---|---:|---:|---:|---:|
| sarima | 1.0680 | 1.2768 | 0.5005 | 114 |
| timesfm | 1.4645 | 1.8987 | 0.6802 | 114 |
| prophet | 2.6241 | 3.5762 | 1.2336 | 114 |

### Comparativos consolidados

Os comparativos entre amostras ficam em:

- `results/comparisons/all_metrics_long.csv`
- `results/comparisons/best_model_by_sample.csv`
- `results/comparisons/smape_by_model_and_sample.csv`
- `results/comparisons/README.md`

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

## Nota sobre execução neste ambiente

Nesta sessão, a instalação com `pip` falhou por restrição de rede para acessar PyPI.
No seu ambiente com internet, execute `make setup-base` (ou `make setup-full`) e depois os alvos de teste.
