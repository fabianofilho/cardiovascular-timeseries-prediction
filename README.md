# Predição de Eventos Cardiovasculares em São Paulo

Análise de séries temporais de mortalidade cardiovascular no estado de São Paulo, usando dados reais do **Sistema de Informações sobre Mortalidade (SIM/DataSUS)**.

## Objetivo

Avaliar a capacidade de modelos de forecasting em prever a tendência mensal de óbitos por doenças cardiovasculares (CID-10 capítulo I: I00-I99) no estado de São Paulo, usando backtesting temporal com rolling origin.

Modelos comparados:

1. **SARIMA** — baseline estatístico com sazonalidade multiplicativa
2. **Prophet** — decomposição de tendência + sazonalidade (Meta)
3. **TimesFM** — foundation model para séries temporais (Google)

## Dados

- **Fonte**: SIM/DataSUS — registros reais de óbito
- **Período**: Janeiro/2019 a Dezembro/2023 (60 meses)
- **Região**: Estado de São Paulo (UF=SP)
- **Filtro**: Causa básica com CID-10 prefixo "I" (doenças do aparelho circulatório)
- **Extração**: Download via HTTP mirror do DataSUS com fallback para FTP direto

### Resumo dos dados

| Indicador | Valor |
|---|---:|
| Total de registros brutos (SIM) | 1.775.800 |
| Registros cardiovasculares (CID I) | 461.950 |
| Pontos na série mensal | 60 |
| Média mensal de óbitos | ~7.700 |
| Mínimo mensal | 5.978 (abr/2020) |
| Máximo mensal | 9.582 (jan/2022) |

## Metodologia

- **Backtesting**: Rolling origin cross-validation
- **Horizonte de previsão**: 6 meses
- **Janela mínima de treino**: 24 meses
- **Métricas**: MAE, RMSE, sMAPE
- **Total de previsões por modelo**: 186 pontos (31 janelas x 6 horizontes)

### Configuração dos modelos

| Modelo | Configuração |
|---|---|
| SARIMA | order=(1,1,1), seasonal_order=(0,1,1,12) |
| Prophet | yearly_seasonality=True, weekly/daily=False |
| TimesFM | google/timesfm-2.5-200m-pytorch, context=512 |

## Resultados

Benchmark com dados reais do SIM (SP, 2019-2023):

| Modelo | MAE | RMSE | sMAPE (%) | n_predictions |
|---|---:|---:|---:|---:|
| **TimesFM** | **586.02** | **765.77** | **7.30** | 186 |
| SARIMA | 603.51 | 787.83 | 7.45 | 186 |
| Prophet | 613.80 | 767.23 | 7.68 | 186 |

- **TimesFM** vence em todas as métricas (MAE, RMSE, sMAPE)
- Com média de ~8.000 óbitos/mês, MAE ~586 representa erro de ~7.3%
- Os três modelos capturam a sazonalidade (pico no inverno: jun-jul)
- TimesFM também produz previsões com range mais conservador [6.664, 9.078] vs valores reais [6.217, 9.582]

### Sazonalidade observada

A série apresenta padrão sazonal claro com pico de mortalidade nos meses de inverno (junho-julho) e vale no verão (fevereiro), consistente com a literatura sobre mortalidade cardiovascular e temperatura.

## Conclusões

1. **TimesFM** (foundation model) é o melhor modelo: menor MAE, RMSE e sMAPE, com previsões mais estáveis
2. **SARIMA** é um baseline competitivo (sMAPE 7.45% vs 7.30%), leve e sem dependências pesadas
3. **Prophet** oferece interpretabilidade adicional (decomposição tendência + sazonalidade), mas ficou em terceiro
4. A diferença entre os três modelos é pequena (~0.4 pp de sMAPE), sugerindo que a série tem boa previsibilidade
5. A escolha final deve considerar: robustez operacional, interpretabilidade e custo computacional

## Próximos passos

1. Extrair SIH/SIM para série mensal por município/DRS
2. Criar variáveis exógenas (temperatura, cobertura APS, envelhecimento)
3. Rodar benchmark por estrato (sexo, faixa etária, subcategoria CID, território)
4. Definir modelo campeão por métrica e robustez operacional

## Referências

- **SIM/DataSUS**: Sistema de Informações sobre Mortalidade — Ministério da Saúde
- **CID-10 Cap. IX**: Doenças do aparelho circulatório (I00-I99)
- Dicionário CID-10 cardiovascular: `docs/cid10_cardiovascular_sim.md`
- Faixas CID automatizadas: `docs/cid10_cardiovascular_ranges.csv`
- Colunas SIH/SIM: `docs/sih_sim_colunas_incidencia.md`
