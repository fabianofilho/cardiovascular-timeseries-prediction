# CLAUDE.md

Este documento define as premissas para executar este repositório e conduzir experimentos de predição de eventos cardiovasculares em São Paulo com SIH/SIM.

## 1) Objetivo e escopo

- Objetivo principal: prever contagens de eventos cardiovasculares em série temporal.
- Fontes-alvo:
  - SIH/SUS: internações por causas cardiovasculares.
  - SIM: óbitos por causas cardiovasculares.
- Unidade temporal padrão: mensal (`MS`, primeiro dia do mês).
- Unidade geográfica inicial: Estado de São Paulo (UF = `SP`).

## 2) Definição de variável-alvo

- A variável `value` deve representar contagem agregada por mês.
- O CSV de entrada do benchmark deve conter, no mínimo:
  - `date`: data do período.
  - `value`: contagem do evento no período.
- O benchmark sempre agrega para a frequência informada em `--freq`.

## 3) Premissas de qualidade dos dados

- Datas inválidas e valores não numéricos são tratados como erro de execução.
- Dados faltantes após agregação mensal são preenchidos com `0.0`.
- Todo experimento deve registrar:
  - fonte (SIH/SIM),
  - recorte geográfico,
  - recorte temporal,
  - regra de seleção de CID,
  - data/hora da extração.

## 4) Ambiente de execução

- Python recomendado: `3.10+`.
- Setup mínimo:
  - `make setup-base`
- Setup completo (inclui modelos opcionais pesados):
  - `make setup-full`
- Dependências:
  - Base em `requirements.txt`.
  - Opcionais em `requirements-optional.txt`.

## 5) Fluxo padrão de experimento

1. Criar/ativar ambiente virtual.
2. Rodar smoke test local com série pequena:
   - `make smoke-baseline`
3. Baixar amostra reduzida com PySUS:
   - `make sample-pysus`
4. Rodar benchmark completo na amostra:
   - `make benchmark-all`
5. Escalar para recorte real SIH/SIM e repetir benchmark.

## 6) Modelos e estratégia de comparação

- Modelos candidatos:
  - `sarima` (baseline obrigatório),
  - `prophet`,
  - `timesfm`.
- Comparação deve ser feita com backtesting temporal (rolling origin).
- Não selecionar modelo por conveniência teórica; selecionar por resultado em validação temporal.

## 7) Métricas de avaliação

- Métricas oficiais do repositório:
  - `MAE`,
  - `RMSE`,
  - `sMAPE`.
- Critério primário sugerido: menor `sMAPE` sem perda relevante de estabilidade.
- Sempre reportar `n_predictions` para contextualizar confiabilidade da métrica.

## 8) Regras para evitar viés e vazamento

- Nunca usar informação futura no treinamento de previsões passadas.
- Não embaralhar a série temporal (sem split aleatório).
- Manter ordem temporal estrita em treino/validação/teste.
- Se usar variáveis exógenas, garantir disponibilidade real no momento de previsão.

## 9) Reprodutibilidade e governança

- Todo experimento deve salvar saídas em `results/` com prefixo explícito.
- Saídas mínimas:
  - `<prefix>_metrics.csv`
  - `<prefix>_predictions.csv`
- Registrar no commit/PR:
  - comando exato executado,
  - versão dos dados,
  - hash do commit.

## 10) Convenções operacionais

- Executar scripts com `PYTHONPATH=src`.
- Preferir `Makefile` para padronizar execução.
- Em caso de indisponibilidade de `prophet` ou `timesfm`, o pipeline deve continuar com os modelos disponíveis.

## 11) Escalonamento recomendado

- Etapa 1: série única SP mensal (amostra pequena).
- Etapa 2: SP completo com maior histórico temporal.
- Etapa 3: estratificação por sexo/faixa etária/CID/território.
- Etapa 4: inclusão de exógenas e comparação incremental de ganho.

## 12) Checklist antes de declarar modelo campeão

- [ ] Baseline `sarima` rodado e documentado.
- [ ] Mesmo horizonte e mesmas janelas para todos os modelos.
- [ ] Métricas comparadas no mesmo conjunto de previsões.
- [ ] Estabilidade avaliada (sem colapso em janelas específicas).
- [ ] Custo operacional e manutenção considerados.

## 13) Limitações atuais conhecidas

- Wrapper de `TimesFM` pode exigir ajuste conforme versão da biblioteca.
- Disponibilidade de rede pode bloquear instalação de dependências no ambiente sandboxado.
- Qualidade final depende criticamente da definição epidemiológica do recorte de CID cardiovascular.

## 14) Próximas extensões sugeridas

- Adicionar script dedicado de construção de coortes SIH/SIM por CID.
- Adicionar configuração centralizada (YAML/TOML) para experimentos.
- Adicionar relatório automático com ranking, gráficos e análise de erro por período.
