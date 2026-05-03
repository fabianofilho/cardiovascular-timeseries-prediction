# TODO - Proximas iteracoes

Baseado na analise comparativa com o projeto dengue-imdc-2026 e nas limitacoes documentadas no paper.
Ordenado por impacto esperado.

---

## Prioridade 1: Ampliar a serie temporal (desbloqueador)

**Por que primeiro:** a serie de 60 meses (2019-2023) e o limitante critico.
Com 168 meses (2010-2023), fine-tuning do TimesFM vira viavel e o backtesting fica muito mais robusto.

- [ ] Extrair SIM/DataSUS para SP, anos 2010-2018
  - Usar `scripts/extract_sim_real.py --uf SP --years 2010-2018`
  - Validar com `scripts/validate_real_dataset.py` antes de rodar qualquer modelo
  - Concatenar com a serie existente (2019-2023) em `data/processed/`
- [ ] Re-rodar benchmark base (SARIMA + Prophet + TimesFM) na serie estendida
  - Comparar metricas: espera-se melhora expressiva no SARIMA (mais dados sazonais)
  - Registrar resultado em `docs/experiments/benchmark_2010_2023_baseline.md`
- [ ] Atualizar `run_real_pipeline.py` para usar `--years 2010-2023` como padrao

---

## Prioridade 2: Adicionar temperatura como variavel exogena

**Por que:** temperatura minima e o principal driver da sazonalidade cardiovascular
(frio eleva pressao arterial e agregacao plaquetaria). Uma variavel, ganho imediato.
O mesmo movimento que o dengue-imdc fez com clima/umidade.

- [ ] Baixar dados de temperatura mensal para SP via ERA5 (CDS API, gratuito) ou INMET
  - Script sugerido: `scripts/fetch_era5_temperature.py`
  - Variaveis: temperatura minima mensal (tm_min) e media (tm_med) para estado de SP
  - Periodo: 2010-2023, mesma granularidade da serie principal
- [ ] Adicionar suporte a exogenas no `run_benchmark.py` (`--exog-csv` opcional)
- [ ] Criar `XGBoostForecaster` e `CatBoostForecaster` em `src/cv_timeseries/models.py`
  - Interface identica a `Forecaster` base
  - Features: lags 1-12 meses + temperatura como exogena
  - Seguir o padrao skforecast/ForecasterRecursive do projeto dengue
- [ ] Rodar benchmark comparativo: boosting com temperatura vs. TimesFM zero-shot
  - Hipotese: XGBoost com temperatura pode chegar perto ou superar TimesFM
  - Registrar em `docs/experiments/benchmark_boosting_temperatura.md`

---

## Prioridade 3: Fine-tuning do TimesFM

**Por que:** zero-shot ja deu 7.3% sMAPE. Fine-tuning em 168 meses de dados
brasileiros de inverno/verao deve baixar mais. Mortalidade CV e mais estavel
que dengue, cenario ideal para fine-tuning.

**Depende do item 1 (serie estendida).**

- [ ] Avaliar se GPU do laboratorio suporta fine-tuning (requer CUDA, ~8GB VRAM)
- [ ] Implementar `TimesFMFineTunedForecaster` em `src/cv_timeseries/models.py`
  - Usar checkpoint `google/timesfm-2.5-200m-pytorch` como base
  - Fine-tuning com janelas de treino do rolling-origin (sem vazar futuro)
- [ ] Comparar zero-shot vs. fine-tuned nas mesmas janelas de backtesting
- [ ] Documentar ganho marginal em `docs/experiments/timesfm_finetune.md`

---

## Prioridade 4: Estratificacao por faixa etaria

**Por que:** adultos 40-59, idosos 60-79 e muito idosos 80+ tem padroes de
mortalidade cardiovascular completamente diferentes. Um modelo por estrato com
ensemble final e muito mais util clinicamente que o agregado estadual.

- [ ] Extrair campo idade do SIM (`IDADE` ou `IDADEanos` no DBC)
  - Criar faixas: 0-39, 40-59, 60-79, 80+
  - Salvar series separadas em `data/processed/serie_faixa_*.csv`
- [ ] Rodar benchmark para cada faixa com os modelos existentes
- [ ] Comparar sazonalidade entre faixas (grafico sobreposto)
- [ ] Ensemble simples: media ponderada por proporcao de obitos por faixa

---

## Prioridade 5: Estratificacao por subcategoria CID-10

**Por que:** IAM, AVC e insuficiencia cardiaca tem drivers distintos.
IAM tem pico mais abrupto no frio. AVC hemorragico e isquemico diferem.
Util para planejamento de UTI coronariana vs. leitos de AVC.

- [ ] Usar mapeamento existente em `docs/cid10_cardiovascular_ranges.csv`
- [ ] Extrair series para: IAM (I21-I22), AVC (I60-I64), IC (I50), outras (resto I00-I99)
- [ ] Rodar SARIMA separado por subcategoria (menos computacional)
- [ ] Documentar perfis sazonais distintos

---

## Prioridade 6: Granularidade semanal

**Por que:** no dengue a migracao mensal -> semanal reduziu sMAPE de 78% para 33%.
Para CV, o ganho esperado e menor (sazonalidade ja e estavel no mensal), mas
pode revelar padroes intra-mensais ligados a ondas de frio.

**Risco:** SIM libera dados com delay; dados semanais recentes podem ser incompletos.

- [ ] Verificar se DataSUS/SIM disponibiliza data de obito com precisao semanal
  - Campo `DTOBITO` no DBC tem resolucao diaria - agregacao semanal e possivel
- [ ] Criar agregacao semanal em `scripts/extract_sim_real.py` (`--freq W`)
- [ ] Ajustar lags no modelo (usar 52 semanas em vez de 12 meses)
- [ ] Comparar sMAPE semanal vs. mensal nas mesmas janelas de backtesting

---

## Infraestrutura / qualidade

- [ ] Adicionar `XGBoost` e `CatBoost` em `requirements-optional.txt`
- [ ] Criar `make benchmark-boosting` no Makefile
- [ ] Atualizar `docs/experiments/` com template padrao de registro de experimento
- [ ] Adicionar script `scripts/plot_stratified_results.py` para figuras por estrato
