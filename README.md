# Predição de Eventos Cardiovasculares em São Paulo

Análise de séries temporais de mortalidade cardiovascular no estado de São Paulo, usando dados reais do **Sistema de Informações sobre Mortalidade (SIM/DataSUS)**.

## Objetivo

Avaliar a capacidade de modelos de forecasting em prever a tendência mensal de óbitos por doenças cardiovasculares (CID-10 capítulo I: I00-I99) no estado de São Paulo, com backtesting temporal por rolling origin e **incerteza medida** em todas as comparações.

Modelos comparados:

1. **SARIMA**: baseline estatístico com sazonalidade multiplicativa
2. **Prophet**: decomposição de tendência e sazonalidade (Meta)
3. **TimesFM**: foundation model para séries temporais (Google), zero-shot
4. **XGBoost** e **CatBoost**: boosting com defasagens, via skforecast

## Dados

- **Fonte**: SIM/DataSUS, registros reais de óbito
- **Período**: Janeiro/2010 a Dezembro/2023 (168 meses)
- **Região**: Estado de São Paulo (UF=SP)
- **Filtro**: Causa básica com CID-10 prefixo "I" (doenças do aparelho circulatório)
- **Extração**: `scripts/extract_sim_real.py`, com validação em `scripts/validate_real_dataset.py` (9 de 9 checagens aprovadas, `results/data_reality_report_2010_2023.json`)

### Resumo dos dados

| Indicador | Valor |
|---|---:|
| Total de registros brutos (SIM) | 4.317.224 |
| Registros cardiovasculares (CID I) | 1.217.427 |
| Pontos na série mensal | 168 |
| Média mensal de óbitos | 7.247 |
| Mínimo mensal | 5.811 (fev/2011) |
| Máximo mensal | 9.582 (jan/2022) |
| Óbitos em 2010 | 79.933 |
| Óbitos em 2023 | 95.538 |

A série agregada está versionada em `results/series/serie_eventos_sp_sim_real_2010_2023.csv`, sem microdado.

## Metodologia

- **Backtesting**: rolling origin com janela expanding, passo 1
- **Horizonte de previsão**: 6 meses
- **Janela mínima de treino**: 60 meses
- **Janelas**: 103, totalizando **618 previsões por modelo**
- **Métricas**: MAE, RMSE, sMAPE

### Incerteza

Comparar modelos por diferença pontual de sMAPE não diz se a ordem observada é sinal ou ruído. Todas as comparações abaixo carregam intervalo de confiança:

- **Bootstrap por bloco**: reamostragem de **janelas inteiras** (não de previsões isoladas, que são dependentes entre si), B = 10.000 réplicas, semente 20260817, as mesmas janelas para todos os modelos em cada réplica, o que preserva o pareamento
- **Diebold-Mariano** por horizonte, com correção de Harvey
- **Critério pré-declarado**: uma melhoria só é considerada confirmada se o IC95% da diferença exclui zero **e** o DM fica abaixo de 0,05 em 3 ou mais dos 6 horizontes

### Configuração dos modelos

| Modelo | Configuração |
|---|---|
| SARIMA | order=(1,1,1), seasonal_order=(0,1,1,12) |
| Prophet | yearly_seasonality=True, weekly/daily=False |
| TimesFM | google/timesfm-2.5-200m-pytorch, contexto 512, zero-shot |
| XGBoost, CatBoost | skforecast ForecasterRecursive, lags 1-12, previsão recursiva |

## Resultados

### Desempenho agregado

| Modelo | MAE | RMSE | sMAPE (%) | IC95% do sMAPE | largura |
|---|---:|---:|---:|:---:|---:|
| Prophet | 355,17 | 515,76 | 4,70 | [4,17; 5,27] | 1,10 pp |
| SARIMA | 363,26 | 508,11 | 4,80 | [4,29; 5,33] | 1,04 pp |
| TimesFM | 366,04 | 513,39 | 4,83 | [4,32; 5,39] | 1,06 pp |
| CatBoost | 500,90 | 673,56 | 6,59 | [5,92; 7,27] | 1,35 pp |
| XGBoost | 526,58 | 704,47 | 6,94 | [6,26; 7,65] | 1,39 pp |

### Não há vencedor entre Prophet, SARIMA e TimesFM

A amplitude entre os três primeiros é de **0,13 pp**, contra largura média de IC de **1,07 pp**. As três diferenças pareadas têm intervalo contendo zero (p de 0,17 a 0,77) e **nenhuma das 18 células de Diebold-Mariano** fica abaixo de 0,05.

O ranking pontual desta rodada é o inverso do da rodada anterior com 60 meses, onde o TimesFM aparecia à frente e o Prophet em último. A inversão com mais dados é a evidência direta de que aquela ordem estava dentro do ruído.

A escolha entre os três é operacional, não estatística: custo computacional, facilidade de manutenção e interpretabilidade. O SARIMA roda em CPU sem dependências pesadas e entrega o mesmo desempenho.

### O boosting é detectavelmente pior

XGBoost e CatBoost com defasagens e sem exógena ficam 1,8 a 2,2 pp acima do grupo da frente, com p_bootstrap = 0,000 e **38 de 42 células** de DM abaixo de 0,05. Aqui a diferença sai do ruído.

### O que move o erro é o tamanho da série, não o modelo

Comparando **as mesmas datas de teste** (2021-2023) contra a rodada de 60 meses:

| Modelo | sMAPE com treino de 24-54 meses | sMAPE com treino de 132-168 meses | ganho |
|---|---:|---:|---:|
| Prophet | 7,68 | 5,97 | 1,71 pp |
| SARIMA | 7,45 | 6,18 | 1,27 pp |
| TimesFM | 7,30 | 6,07 | 1,23 pp |

Nove anos a mais de treino valem de 1,2 a 1,7 pp, uma ordem de grandeza acima dos 0,38 pp que separavam os modelos na rodada anterior. Estender a série entregou mais do que qualquer troca de modelo entregou.

O sMAPE agregado de 4,7 a 4,8% não é comparável com os 7,3% da rodada anterior, porque o período de teste mudou junto. A comparação válida é a da tabela acima.

### Onde o TimesFM tem vantagem: histórico curto

Repetindo o benchmark nas mesmas 103 origens, mas com o treino limitado aos últimos 60 meses (janela deslizante):

| Modelo | expanding | deslizante 60 | diferença |
|---|---:|---:|---:|
| TimesFM | 4,83 | 4,74 | indiferente (p=0,42) |
| Prophet | 4,70 | 5,39 | perde 0,69 pp |
| SARIMA | 4,80 | 5,46 | perde 0,66 pp (confirmado) |

Descartar o histórico além de 5 anos custa caro a SARIMA e Prophet, e **não afeta o TimesFM**, cujo conhecimento vem do pré-treino e não da série alvo. No regime de histórico curto o TimesFM lidera, com DM abaixo de 0,05 contra o SARIMA em h3, h4 e h5.

Esta é a caracterização defensável do papel do foundation model: ele não é mais preciso quando há histórico longo, ele é **robusto quando não há**.

### Temperatura como exógena

Temperatura mínima mensal do INMET (estação A701, Mirante de Santana), com política anti-vazamento: o valor futuro é a climatologia do mês do ano recalculada por janela e truncada no fim de cada treino.

| Modelo | sem exógena | com temperatura | ganho | IC95% da diferença | DM < 0,05 |
|---|---:|---:|---:|:---:|:---:|
| SARIMA | 4,80 | 4,66 | 0,14 pp | [0,034; 0,257] | 1 de 6 |
| CatBoost | 6,59 | 6,33 | 0,26 pp | [0,073; 0,452] | 1 de 6 |
| XGBoost | 6,94 | 6,55 | 0,38 pp | [0,222; 0,543] | 2 de 6 |

Os três ganhos têm IC excluindo zero, mas **nenhum atinge o critério pré-declarado** de DM em 3 ou mais horizontes. O resultado é promissor e não confirmado.

Um cenário-teto com a temperatura futura observada (vazamento deliberado e rotulado) adiciona apenas 0,03 pp sobre a climatologia. Ou seja, **o conteúdo preditivo da temperatura mensal é a climatologia**: saber a temperatura futura de verdade quase não acrescenta ao que a média histórica do mês já dá.

### Sazonalidade

Padrão sazonal estável ao longo dos 14 anos, com pico em julho (média de 8.247 óbitos) e vale em fevereiro (6.273), amplitude de 27,2% em torno da média. Consistente com a literatura sobre temperatura e mortalidade cardiovascular.

## Conclusões

1. Com incerteza medida, **Prophet, SARIMA e TimesFM empatam** nas duas extensões de série testadas. Não há base estatística para eleger um vencedor, e o modelo de menor custo computacional atende ao uso.
2. O ganho real veio de **estender a série**, não de trocar de modelo: 1,2 a 1,7 pp contra 0,13 pp de amplitude entre modelos.
3. O foundation model se justifica pela **robustez a histórico curto**, não por precisão superior. É o cenário de quem quer prever muitos indicadores ou territórios sem série longa em cada um.
4. Boosting com defasagens e sem exógena é pior, com significância em todos os horizontes.
5. Temperatura ajuda, mas abaixo do critério de confirmação, e o que ela carrega é essencialmente climatologia.

## Reprodução

```bash
make setup-full
```

Benchmark principal:

```bash
PYTHONPATH=src python scripts/run_benchmark.py --input-csv results/series/serie_eventos_sp_sim_real_2010_2023.csv --horizon 6 --min-train-size 60 --models sarima,prophet,timesfm,xgboost,catboost --output-prefix results/benchmark_sim_real_sp_2010_2023
```

Incerteza:

```bash
PYTHONPATH=src python scripts/run_uncertainty.py --predictions-csv results/benchmark_sim_real_sp_2010_2023_predictions.csv --models timesfm,sarima,prophet,xgboost,catboost --seed 20260817 --output-prefix results/uncertainty_2010_2023
```

## Manuscrito

O manuscrito está em [`paper/manuscript.tex`](paper/manuscript.tex), com tabelas, figuras e o JSON de conferência gerados dos dados brutos por um script só:

```bash
PYTHONPATH=src python scripts/build_paper_assets.py
cd paper && tectonic -X compile manuscript.tex --outdir ../build
```

Nenhum número entra no manuscrito por digitação: todo valor citado na prosa sai de `paper/verified_numbers.json`, emitido pelo mesmo gerador que produz as tabelas e as figuras. As figuras são código pgfplots, não binário, então o manuscrito cabe num `.tex` único e pode ser movido para qualquer editor.

Para os coautores revisarem, com figuras embutidas no ponto do texto e tabelas nativas do Word:

```bash
python scripts/build_docx.py
```

## Próximos passos

1. Fine-tuning do TimesFM na série estendida (pendente de GPU)
2. Modelar taxa por população residente, em vez de contagem, para separar demografia de tendência epidemiológica
3. Benchmark por estrato: faixa etária, sexo, subcategoria CID (IAM, AVC, IC), território
4. Granularidade semanal, para captar ondas de frio

## Documentação

- Experimentos detalhados: `docs/experiments/`
- Task graphs das rodadas: `task-graph.md`, `task-graph-2010-2023.md`, `graphs/`
- Dicionário CID-10 cardiovascular: `docs/cid10_cardiovascular_sim.md`
- Faixas CID automatizadas: `docs/cid10_cardiovascular_ranges.csv`
- Colunas SIH/SIM: `docs/sih_sim_colunas_incidencia.md`

## Referências

- **SIM/DataSUS**: Sistema de Informações sobre Mortalidade, Ministério da Saúde
- **CID-10 Cap. IX**: Doenças do aparelho circulatório (I00-I99)
- **INMET**: Instituto Nacional de Meteorologia, dados históricos horários
