# Dicionário CID-10 Cardiovascular (SIM)

Este arquivo define o dicionário de CIDs para identificar óbitos por doenças cardiovasculares no SIM.

## Regra operacional principal

- Considerar como cardiovascular todo registro com `CAUSABAS` no intervalo **CID-10 `I00` a `I99`**.
- Em termos práticos: `CAUSABAS` iniciando por letra `I`.

## Capítulo CID-10 utilizado

- **Capítulo IX (I00-I99): Doenças do aparelho circulatório**.

## Subgrupos relevantes para análise

| Faixa CID-10 | Grupo |
|---|---|
| I00-I02 | Febre reumática aguda |
| I05-I09 | Doenças cardíacas reumáticas crônicas |
| I10-I15 | Doenças hipertensivas |
| I20-I25 | Doenças isquêmicas do coração |
| I26-I28 | Doença cardíaca pulmonar e da circulação pulmonar |
| I30-I52 | Outras formas de doença do coração |
| I60-I69 | Doenças cerebrovasculares |
| I70-I79 | Doenças das artérias, arteríolas e capilares |
| I80-I89 | Doenças das veias, vasos linfáticos e linfonodos |
| I95-I99 | Outros transtornos do aparelho circulatório |

## Exemplos frequentes de códigos-alvo

- `I10` Hipertensão essencial
- `I21` Infarto agudo do miocárdio
- `I25` Doença isquêmica crônica do coração
- `I48` Fibrilação/Flutter atrial
- `I50` Insuficiência cardíaca
- `I61` Hemorragia intracerebral
- `I63` Infarto cerebral
- `I64` AVC não especificado

## Recomendação para coorte principal

- Coorte ampla: `CAUSABAS` em `I00-I99`.
- Análises específicas: estratificar por subgrupos (ex.: `I20-I25`, `I60-I69`, `I10-I15`).

## Observações

- `CAUSABAS` representa a causa básica do óbito no SIM e deve ser o campo principal para incidência de mortalidade cardiovascular.
- Se houver interesse em causas associadas, usar campos de causas múltiplas em análise complementar, sem substituir a regra principal por `CAUSABAS`.
