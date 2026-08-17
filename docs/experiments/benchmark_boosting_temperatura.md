# Benchmark Temperatura como Exógena (Prioridade 2 do TODO)

Data de execução: 2026-08-17

## Dados de temperatura

- Fonte: INMET/BDMEP, zips anuais públicos, estação automática **A701 São Paulo – Mirante de Santana** (fallbacks A755/A711 previstos e **não usados** — a A701 cobriu os 168 meses sem buraco e sem interpolação)
- Script: `scripts/fetch_temperature_inmet.py`; gates de qualidade: dia com ≥18 horas válidas, mês com ≥20 dias válidos, máximo 2 meses interpoláveis
- Série: `results/series/temperatura_sp_mensal_2010_2023.csv` (`date,tmin,tmed`); variável usada: `tmin` (média mensal das mínimas diárias — frio eleva PA e agregação plaquetária, hipótese do TODO)

## Desenho anti-vazamento (o ponto central da rodada)

Para cada janela com treino até `t`, prevendo `t+1..t+6`:

- `exog_train` = temperatura **observada** até `t` (sempre legítimo).
- `exog_future` por política (`--exog-policy` em `run_benchmark.py`):
  - **`climatology` (principal)** — média do mês-do-ano calculada exclusivamente sobre `[início, t]`, recalculada a cada janela (expanding). Zero vazamento; é o cenário operacional real: em produção não se conhece a temperatura futura, mas conhece-se a climatologia.
  - **`lag12`** — temperatura observada em `mês−12`. Sem vazamento; análise de sensibilidade (não rodada aqui, disponível na flag).
  - **`observed`** — temperatura futura real. **Vazamento deliberado e rotulado**: cenário-teto equivalente a previsão meteorológica perfeita. Reportado só na seção própria abaixo, nunca comparável ao benchmark principal.
- Regra dura no código: para climatology/lag12 o futuro é montado a partir da série **truncada em `t`**, nunca do CSV completo (`build_exog_frames` em `scripts/run_benchmark.py`).
- Modelos com exógena: SARIMAX (`exog=` nativo), XGBoost/CatBoost (skforecast `fit/predict(exog=)`). Prophet e TimesFM ficam sem exógena por desenho. Sufixo `_temp` no nome evita colisão com os baselines.

## Critérios de decisão (declarados antes de olhar o resultado)

1. IC95% bootstrap da diferença pareada de sMAPE exclui zero (p < 0,05);
2. Diebold-Mariano (Harvey) p < 0,05 em ≥3 dos 6 horizontes, na mesma direção;
3. Razão amplitude/largura-de-IC > 1 para ranking informativo.

Melhoria **confirmada** = (1) e (2). Só (1) = **sugestiva**. Nenhum = empate.

## Resultados — política `climatology` (principal)

Mesmas 103 janelas do baseline (pareamento perfeito, n=618 por modelo). Arquivos: `results/benchmark_exog_temp_climatology_*.csv`, `results/uncertainty_exog_climatology_*.csv`.

| model | sMAPE (%) | vs baseline (pp) | IC95% da diferença | p_bootstrap | DM < 0,05 |
|---|---:|---:|---:|---:|---|
| sarima_temp | 4.66 | −0.14 | [−0.257, −0.034] | 0.006 | h1 |
| catboost_temp | 6.33 | −0.26 | [−0.452, −0.073] | 0.006 | h6 |
| xgboost_temp | 6.55 | −0.38 | [−0.543, −0.222] | 0.000 | h1, h6 |

- A temperatura melhora **os três modelos** com IC pareado excluindo zero — mas nenhum passa no critério 2 (≥3 horizontes). Veredito pelos critérios pré-declarados: **melhoria sugestiva, não confirmada**.
- A melhoria é **dentro do modelo**, não entre famílias: `sarima_temp` (4,66) vs `prophet` sem exógena (4,70) dá diferença 0,04 pp, IC [−0,266, 0,334], p=0,77 — o topo da tabela continua empatado.
- Boosting com temperatura continua atrás do top-3 sem exógena (6,33–6,55 vs 4,70–4,83). A hipótese do TODO ("XGBoost com temperatura pode chegar perto ou superar TimesFM") **não se sustentou** nesta série.

## Cenário-teto — política `observed` (vazamento rotulado)

| model | sMAPE teto (%) | sMAPE climatology (%) | ganho extra do teto (pp) |
|---|---:|---:|---:|
| sarima_temp | 4.63 | 4.66 | 0.03 |
| catboost_temp | 6.30 | 6.33 | 0.03 |
| xgboost_temp | 6.55 | 6.55 | 0.00 |

Conhecer a temperatura futura real acrescenta ~0,03 pp sobre a climatologia: **o valor preditivo da temperatura está quase todo na sua sazonalidade média**, que os modelos sazonais já capturam por outros meios. Isso explica o ganho pequeno e é um resultado em si.

## Conclusão rápida

1. Temperatura como exógena, sem nenhum vazamento, dá ganho real porém pequeno (0,14–0,38 pp) e abaixo do critério de confirmação.
2. O teto com vazamento deliberado mostra que não há mais o que extrair da temperatura mensal agregada — o sinal é a climatologia.
3. Caminhos com mais potencial que refinar a exógena: granularidade semanal (Prioridade 6, onde a anomalia de temperatura semanal pode ter sinal que o mês médio apaga) e fine-tuning do TimesFM (Prioridade 3, agora desbloqueada pela série de 168 meses; adiada nesta rodada por falta de GPU ≥8GB nesta máquina — um fine-tune por janela em CPU é inviável).
