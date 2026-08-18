# TODO: proximas iteracoes

Ordenado por impacto esperado. A ordem abaixo mudou depois da rodada de 17/08/2026:
com incerteza medida, a diferenca entre familias de modelo ficou em 0,13 pp, enquanto
estender a serie valeu de 1,2 a 1,7 pp nas mesmas datas de teste. **O que move o erro e
o dado, nao a arquitetura.** As prioridades refletem isso.

---

## Concluido

### ~~Prioridade 1: ampliar a serie temporal~~ (feito no PR #2, 17/08/2026)

- [x] Extrair SIM/DataSUS para SP, 2010-2018, e concatenar com 2019-2023
  - 168 meses, 4.317.224 registros brutos, 1.217.427 obitos CV
  - Validador 9/9 em `results/data_reality_report_2010_2023.json`
  - Reconciliacao com a rodada anterior: os 36 meses de 2021-2023 batem com divergencia 0,000%
- [x] Re-rodar o benchmark base na serie estendida
  - 103 janelas x 6 = 618 previsoes por modelo, min_train=60
  - Registrado em `docs/experiments/benchmark_2010_2023_baseline.md`
  - Resultado: os IC cairam de 2,45 para 1,07 pp de largura, e o ranking do top-3 **inverteu**,
    confirmando que a ordem anterior era ruido
- [x] Serie agregada versionada em `results/series/` (sem microdado)

### ~~Prioridade 2: temperatura como variavel exogena~~ (feito no PR #2, 17/08/2026)

- [x] Baixar temperatura mensal para SP via INMET (`scripts/fetch_temperature_inmet.py`)
  - Estacao A701 (Mirante de Santana), tmin e tmed, 168 meses sem buraco
  - ERA5 nao foi necessario
- [x] Suporte a exogenas em `run_benchmark.py` (`--exog-csv`, `--exog-cols`, `--exog-policy`)
  - Tres politicas: `climatology` (anti-vazamento, usada na comparacao), `lag12`, `observed`
    (vazamento deliberado e rotulado, cenario-teto)
- [x] `XGBoostForecaster` e `CatBoostForecaster` em `src/cv_timeseries/models.py`
- [x] Benchmark comparativo boosting com temperatura vs. top-3
  - Registrado em `docs/experiments/benchmark_boosting_temperatura.md`
  - **Hipotese refutada**: boosting com temperatura (6,33 a 6,55) nao chega perto do top-3 (4,7 a 4,8).
    Sem exogena o boosting e pior com DM < 0,05 em 6 de 6 horizontes
  - Temperatura reduz o erro dos 3 modelos que a recebem (0,14 a 0,38 pp, IC excluem zero), mas
    **abaixo do criterio pre-declarado** (DM em 3 ou mais de 6 horizontes). Nao confirmado
  - O cenario-teto adiciona so 0,03 pp sobre a climatologia: o conteudo preditivo da temperatura
    mensal e a climatologia

### ~~Corrigir README e paper~~ (LAB-64, feito em 17/08/2026)

- [x] README e `paper.md` reescritos com a redacao que os dados sustentam, sem eleicao de vencedor

---

## Prioridade 1: modelar taxa em vez de contagem

**Por que primeiro:** a serie e de contagens absolutas, entao parte da tendencia que o historico
longo captura e crescimento e envelhecimento populacional, nao mudanca epidemiologica. Os obitos
anuais subiram de 79.933 (2010) para 95.538 (2023), e nada no desenho atual separa demografia de
risco. Este e o limite explicito registrado em `docs/experiments/benchmark_window_sensitivity.md`.

- [ ] Obter populacao residente de SP por ano (IBGE, estimativas intercensitarias 2010-2023)
- [ ] Converter a serie para taxa por 100.000 habitantes, mantendo a serie de contagem ao lado
- [ ] Re-rodar o benchmark nas duas escalas e comparar
  - Hipotese: a vantagem do historico longo cai, porque parte dela era demografia
- [ ] Registrar em `docs/experiments/benchmark_taxa_vs_contagem.md`

---

## Prioridade 2: estratificacao por faixa etaria

**Por que:** adultos 40-59, idosos 60-79 e muito idosos 80+ tem padroes de mortalidade
cardiovascular diferentes. Um modelo por estrato com ensemble final e mais util clinicamente
que o agregado estadual. Tambem e o cenario que testa a vantagem real do TimesFM, ja que cada
estrato tem serie mais curta que o agregado.

- [ ] Extrair campo idade do SIM (`IDADE` ou `IDADEanos` no DBC)
  - Faixas: 0-39, 40-59, 60-79, 80+
- [ ] Rodar benchmark por faixa, com incerteza, mesmos criterios pre-declarados
- [ ] Comparar sazonalidade entre faixas (grafico sobreposto)
- [ ] Ensemble por media ponderada pela proporcao de obitos de cada faixa

---

## Prioridade 3: granularidade semanal

**Por que:** e a condicao sob a qual a temperatura pode passar do criterio de confirmacao.
No mensal, o conteudo preditivo dela e a climatologia, porque ondas de frio duram dias e somem
na media do mes. No dengue, a migracao mensal para semanal reduziu o sMAPE de 78% para 33%.

**Risco:** o SIM libera dados com atraso; semanas recentes podem estar incompletas.

- [ ] Agregacao semanal em `scripts/extract_sim_real.py` (`--freq W`), usando `DTOBITO` (resolucao diaria)
- [ ] Ajustar lags do boosting (52 semanas em vez de 12 meses) e o contexto do TimesFM
- [ ] Temperatura semanal do INMET, com as mesmas tres politicas de exogena
- [ ] Comparar sMAPE semanal vs. mensal nas mesmas janelas, e reavaliar o criterio de confirmacao

---

## Prioridade 4: fine-tuning do TimesFM

**Por que:** o pre-requisito de serie esta cumprido (168 meses). Mas a expectativa mudou: o
zero-shot **empata** com os classicos no historico longo, entao o ganho a buscar nao e superar
o SARIMA no agregado, e sim ampliar a vantagem que ele ja tem no regime de historico curto.

**Bloqueado:** requer GPU com ~8GB VRAM, indisponivel na maquina atual.

- [ ] Avaliar disponibilidade de GPU no laboratorio
- [ ] Implementar `TimesFMFineTunedForecaster` em `src/cv_timeseries/models.py`
  - Fine-tuning apenas com as janelas de treino do rolling origin, sem vazar futuro
- [ ] Comparar zero-shot vs. fine-tuned nas mesmas janelas, com IC
- [ ] Testar tambem no regime de janela deslizante de 60 meses, que e onde o modelo se destaca
- [ ] Documentar em `docs/experiments/timesfm_finetune.md`

---

## Prioridade 5: estratificacao por subcategoria CID-10

**Por que:** IAM, AVC e insuficiencia cardiaca tem drivers distintos. IAM tem pico mais abrupto
no frio; AVC hemorragico e isquemico diferem. Util para planejar UTI coronariana vs. leitos de AVC.

- [ ] Usar o mapeamento em `docs/cid10_cardiovascular_ranges.csv`
- [ ] Series para IAM (I21-I22), AVC (I60-I64), IC (I50), outras (resto de I00-I99)
- [ ] Rodar SARIMA por subcategoria (mais barato que o benchmark completo)
- [ ] Documentar perfis sazonais distintos

---

## Prioridade 6: granularidade espacial

**Por que:** a agregacao estadual esconde heterogeneidade entre municipios e DRS. Junto com a
Prioridade 2, e o cenario onde a robustez do foundation model a historico curto seria testada
em escala, com centenas de series curtas em vez de uma longa.

- [ ] Serie por DRS (17 regionais) e depois por municipio para os maiores
- [ ] Benchmark com incerteza, comparando TimesFM zero-shot contra SARIMA por unidade
- [ ] Verificar se a vantagem do TimesFM cresce conforme a serie por unidade encurta

---

## Infraestrutura e qualidade

- [x] XGBoost e CatBoost em `requirements-optional.txt`
- [x] Guarda em `lag12` para horizonte maior que 12 (falha explicita em vez de KeyError cru)
- [x] `run_uncertainty.py` deixou de depender da ordem dos modelos nos IC por horizonte
- [x] Aposentar os geradores legados de figura: `generate_paper_figures.py` e
      `generate_figures_2010_2023.py` sairam de `scripts/` para `images/legacy/`, ao lado das
      figuras que produzem. Ficam versionados porque o registro do defeito cita
      `generate_paper_figures.py:88-116` como evidencia; a numeracao de linha foi preservada
      para a citacao continuar valida. Sao a unica coisa no repo que ainda pede `matplotlib`,
      que por isso segue fora do `requirements.txt` do pipeline
- [ ] Alvos no Makefile para as rodadas novas: `benchmark-2010-2023`, `benchmark-exog`, `uncertainty`
- [ ] Template padrao de registro de experimento em `docs/experiments/`
- [x] Testes automatizados para `build_exog_frames`: 18 testes em `tests/test_exog.py`, com a
      invariante anti-vazamento exercitada nas 103 janelas do desenho real. A funcao saiu de
      `scripts/run_benchmark.py` para `src/cv_timeseries/exog.py` para o teste rodar sem
      statsmodels, prophet nem timesfm. Suite validada por mutacao: quebrar a truncagem da
      climatologia ou remover a guarda do lag12 faz a suite falhar
- [x] Testes de `rolling_origin_splits` e das metricas: 28 testes em `tests/test_evaluate.py`,
      incluindo a invariante de que expanding e deslizante produzem AS MESMAS origens, que e
      pre-requisito do experimento de sensibilidade de janela. Suite total: 46 testes
- [ ] Estender a suite para `load_and_aggregate_series` (`data.py`) e para o bootstrap por
      bloco de `run_uncertainty.py`, que hoje so tem conferencia manual
- [ ] `scripts/plot_stratified_results.py` para figuras por estrato
- [x] Decidir o destino das figuras antigas `fig1_time_series.png` e `fig7_seasonal_profile.png`,
      que contem 24 meses sintetizados: movidas para `images/legacy/`, com README explicando o
      defeito e dizendo para nao reutilizar. Ficam versionadas porque os documentos historicos
      apontam para elas
