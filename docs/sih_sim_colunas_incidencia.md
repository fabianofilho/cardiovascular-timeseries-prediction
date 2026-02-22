# Colunas para Incidência Cardiovascular (SIH e SIM)

Este documento resume quais colunas usar para classificar incidência cardiovascular e montar a série temporal.

## 1) SIM (óbitos)

### Campo obrigatório para classificação

- `CAUSABAS` (causa básica do óbito)

### Regra de incidência cardiovascular

- Incidência cardiovascular = `CAUSABAS` entre `I00` e `I99` (prefixo `I`).

### Campos mínimos recomendados para série mensal

- `CAUSABAS`: filtro CID cardiovascular
- Campo de data do óbito (varia por layout, ex.: `DTOBITO`/`DT_OBITO`)
- Campo geográfico (ex.: UF/município de residência)
- Sexo e idade (opcional para estratificação)

### Exemplo de regra

- `is_cv_sim = CAUSABAS.str.upper().str.startswith("I")`

## 2) SIH (internações)

### Campo obrigatório para classificação principal

- `DIAG_PRINC` (diagnóstico principal da AIH)

### Regra de incidência cardiovascular

- Incidência cardiovascular principal = `DIAG_PRINC` entre `I00` e `I99` (prefixo `I`).

### Campos recomendados para enriquecer análise

- `DIAG_PRINC`: definição da incidência principal
- `DIAG_SECUN` (quando disponível): análise complementar de comorbidades
- Campo de data de internação/competência (ex.: `DT_INTER`, `ANO_CMPT` + `MES_CMPT`)
- Campos geográficos de residência/atendimento
- Sexo e idade

### Exemplo de regra

- `is_cv_sih = DIAG_PRINC.str.upper().str.startswith("I")`

## 3) Padrão de construção da série

1. Filtrar registros cardiovasculares com as regras acima.
2. Converter data para mês de referência (`MS`).
3. Agregar contagem por mês (`value`).
4. Salvar em `date,value` para benchmark.

## 4) Definição recomendada para o projeto

- Mortalidade cardiovascular (SIM): `CAUSABAS` em `I00-I99`.
- Morbidade hospitalar cardiovascular (SIH): `DIAG_PRINC` em `I00-I99`.
- Em análises secundárias, permitir subgrupos CID (IAM, AVC, hipertensivas, etc.).

## 5) Cuidados de consistência

- Padronizar CID em caixa alta e sem espaços.
- Verificar presença de caracteres não alfanuméricos antes do filtro.
- Documentar claramente se a análise usa causa básica (SIM) ou diagnóstico principal (SIH).
