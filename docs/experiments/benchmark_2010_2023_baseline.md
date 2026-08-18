# Benchmark Série Estendida 2010–2023 (baseline, sem exógena)

Data de execução: 2026-08-17

## Configuração

- Série: `results/series/serie_eventos_sp_sim_real_2010_2023.csv` (168 meses, SIM/DataSUS SP, CID-10 prefixo I, validador 9/9 em `results/data_reality_report_2010_2023.json`)
- Frequência: mensal (`MS`)
- Horizonte: `6`
- Janela mínima de treino: `60` (5 anos; ≥4 ciclos sazonais efetivos após a dupla diferenciação do SARIMA. Nota: a rodada 2019–2023 usou 24 no README/paper e 36 no default do script. A divergência fica registrada aqui; README/paper dependem da decisão de LAB-64)
- Janelas: rolling origin expanding, passo 1, gap 0 → **103 janelas × 6 = 618 previsões por modelo** (3,3× as 186 da rodada anterior)
- Modelos: `sarima, prophet, timesfm, xgboost, catboost`, sem exógenas
- TimesFM: `google/timesfm-2.5-200m-pytorch`, zero-shot, CPU
- Mudança de protocolo: o clamp `[0.1·min, 3·max]`, presente só em SARIMA e boosting, foi **removido**; Prophet e TimesFM nunca o tiveram, e a assimetria contaminava a comparação. Nenhuma previsão desta rodada disparou o warning diagnóstico que o substituiu (0 ocorrências no log); o efeito da remoção nesta série foi nulo.
- Reconciliação: os 36 meses 2021–2023 da série nova batem com o `y_true` do benchmark 2019–2023 com divergência 0,000%.

## Métricas com incerteza

Arquivos: `results/benchmark_sim_real_sp_2010_2023_{metrics,predictions}.csv`, `results/uncertainty_2010_2023_*.csv` (B=10.000, bootstrap reamostrando as 103 janelas inteiras, seed 20260817)

| model | mae | rmse | sMAPE (%) | IC95% | largura |
|---|---:|---:|---:|---:|---:|
| prophet | 355.17 | 515.76 | 4.70 | [4.17, 5.27] | 1.10 |
| sarima | 363.26 | 508.11 | 4.80 | [4.29, 5.33] | 1.04 |
| timesfm | 366.04 | 513.39 | 4.83 | [4.32, 5.39] | 1.06 |
| catboost | 500.90 | 673.56 | 6.59 | [5.92, 7.27] | 1.35 |
| xgboost | 526.58 | 704.47 | 6.94 | [6.26, 7.65] | 1.39 |

- Amplitude entre modelos: 2,23 pp; largura média do IC: 1,19 pp; razão 1,88 → ranking informativo **no agregado**, mas dirigido inteiramente pela separação entre boosting e o resto.
- **Top-3 segue em empate técnico**: as três diferenças pareadas entre prophet/sarima/timesfm têm IC contendo zero (p de 0,17 a 0,77) e nenhum Diebold-Mariano abaixo de 0,05 nos 6 horizontes. O ranking pontual **inverteu** em relação à rodada anterior (Prophet passou de último a primeiro), o que confirma que a ordem de 2019–2023 estava dentro do ruído.
- **Boosting sem exógena é detectavelmente pior**: as seis comparações contra o top-3 têm p_bootstrap = 0,000 e DM < 0,05 em todos os 6 horizontes. Com IC medido, a impressão da rodada anterior ("boosting perdeu") agora é afirmável.

## Recorte comparável com a rodada 2019–2023

Mesmas datas de teste (2021-01 a 2023-12; 201 previsões/modelo no recorte novo vs 186 na rodada antiga):

| model | sMAPE antiga (treino 24–54m) | sMAPE nova (treino 132–168m) | ganho (pp) |
|---|---:|---:|---:|
| prophet | 7.68 | 5.97 | 1.71 |
| sarima | 7.45 | 6.18 | 1.27 |
| timesfm | 7.30 | 6.07 | 1.23 |

Nove anos a mais de treino reduzem o sMAPE em 1,2 a 1,7 pp nas mesmas datas, acima da amplitude entre modelos da rodada anterior (0,38 pp). O sMAPE agregado de 4,7 a 4,8% não é comparável com os 7,3% da rodada anterior porque o período 2015–2019 tem erro menor que 2021–2023.

## Conclusão rápida

1. A Prioridade 1 do TODO (estender a série) entregou o que prometia: ICs caíram de ~2,45 pp para ~1,05 pp de largura e o erro caiu 1,2–1,7 pp nas mesmas datas de teste.
2. Entre Prophet, SARIMA e TimesFM **não há vencedor**, agora com intervalos 2,3 vezes menores. A escolha entre eles é operacional (custo computacional, manutenção, interpretabilidade), não estatística.
3. XGBoost/CatBoost com defasagens e sem exógena têm erro maior com significância em todos os horizontes, o que fecha a pergunta de LAB-63 para o caso univariado. Ver `benchmark_boosting_temperatura.md` para o caso com temperatura.

## Achados que exigem decisão

- **Figuras do paper**: `images/legacy/fig1_time_series.png` e `images/legacy/fig7_seasonal_profile.png` da rodada anterior contêm 24 meses sintetizados (2019–2020, `generate_paper_figures.py:88-116`) rotulados como observação. As substitutas com dado 100% observado estão em `images/legacy/fig*_2010_2023.png` (`scripts/generate_figures_2010_2023.py`). Adotar nas próximas versões de README/paper.
- **Versionamento da série (LAB-62)**: a série agregada 2010–2023 está versionada em `results/series/` (168 linhas `date,value`, sem microdado). Mover para `data/processed/` com exceção no `.gitignore` é decisão do dono do repo.
- **min_train_size**: padronizado em 60 nesta rodada; README/paper ainda descrevem 24 (LAB-64).
