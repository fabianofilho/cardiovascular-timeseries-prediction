# Rodada de revisão

Autoria: **Victor Ovani Marchetti** (victorovanimarchetti@usp.br), agosto de 2026.

Este material existia apenas numa pasta do Drive de uma pessoa. Está aqui para deixar de
depender disso. Os resultados correspondentes estão em `results/revisao/`.

## O que cada script faz

| Script | Pergunta que responde |
|---|---|
| `run_optuna.py` | O desempenho fraco dos boosters é falta de ajuste de hiperparâmetros? |
| `run_intervals.py` | SARIMA e Prophet produzem intervalo de previsão; ele é calibrado? |
| `variants.py`, `run_variants.py` | Engenharia de atributos resolve o que o tuning não resolveu? |

## `run_optuna.py`

Dois modos, e a distinção entre eles é o ponto do script:

- `dev`: sem vazamento. Otimiza num backtest interno restrito aos 60 primeiros meses, que
  são treino de todas as 103 janelas do benchmark. Nenhuma data de teste é tocada.
- `oracle`: com vazamento, rotulado. Otimiza direto no sMAPE das janelas de teste. Não é
  resultado reportável, é o teto do que qualquer busca poderia entregar nesta série.

O espaço de busca inclui o número de lags, que no paper está fixo em 12 sem justificativa.

### Reproduzido

Rodamos os hiperparâmetros vencedores do modo `dev` no backtest deste repositório, sob
xgboost 3.2.0:

| Modelo | Reportado por ele | Nossa reprodução | Divergência |
|---|---:|---:|---:|
| CatBoost | 6.4162903089 | 6.4162903089 | 1.7e-11 |
| XGBoost | 7.0648751870 | 7.0648751870 | 1.7e-11 |

Divergência da ordem de 1e-11 é ruído de ponto flutuante. O estudo está correto.

Conclusão: o ajuste melhora o CatBoost em 0,17 pp (6.59 para 6.42) e **piora** o XGBoost em
0,23 pp (6.83 para 7.06). Nem o teto com vazamento do CatBoost (6.35) alcança o naive
sazonal (6.27). Não é falta de busca, é limite do modelo nesta série.

### Sobre a divergência que parecia existir

Os campos `baseline_reproduzida = 6.8535` e `baseline_esperada = 6.8324` em
`results/revisao/ceiling_xgboost.json` pareciam um erro. Não são: correspondem
respectivamente ao xgboost 2.x e ao xgboost 3.0 a 3.2, medidos aqui. Ele tinha dois
ambientes. Ver `docs/xgboost_reprodutibilidade.md`.

## `run_intervals.py`

Mede PICP (cobertura empírica), MPIW (largura média) e o interval score de
Gneiting-Raftery, por horizonte, a 95% nominal, nas mesmas 103 janelas.

### Reproduzido

Reimplementado de forma independente em `scripts/run_calibracao.py`, sem rodar o script
dele, e com uma checagem que o original não faz: se a previsão pontual gerada junto com o
intervalo é a mesma já guardada no benchmark.

| | Ele reporta | Nossa reprodução |
|---|---:|---:|
| SARIMA PICP | 84,5% | 84,5% |
| Prophet PICP | 77,0% | 77,2% |

O SARIMA bate exato. A diferença do Prophet não é erro de ninguém: a banda dele vem de
amostragem posterior não semeada e muda a cada execução. Com semente fixa passa a ser
reprodutível. Ver o apêndice de `docs/xgboost_reprodutibilidade.md`.

Conclusão: os dois subcobrem, o SARIMA por 10,5 pp e o Prophet por 17,8 pp. E o Prophet, que
tem o melhor sMAPE dos cinco, tem o pior dos dois intervalos. Está no manuscrito, na subseção
"The prediction intervals are not calibrated".

## `variants.py` e `run_variants.py`

Seis variantes, mudando uma coisa de cada vez: `base` (o do paper), `roll` (estatísticas de
janela), `cal` (Fourier sobre mês do ano), `diff12` (alvo em diferença sazonal), `full` e
`direct` (um modelo por horizonte).

### Reproduzido

Rodei a grade inteira. Os hiperparâmetros do `base` dele são os mesmos de
`src/cv_timeseries/models.py`, então o `base` funciona como controle da bancada, e passou:

| controle | esperado | obtido |
|---|---:|---:|
| `catboost_base` | 6,5893 | **6,5893** |
| `xgboost_base` | 6,8324 (xgboost 3.2.0) | **6,8324** |

E o número dele reproduz: `catboost_direct` = **6,0921**.

### Mas o 6,09 não é o que parece

Submetido ao critério pré-declarado do paper (IC do bootstrap pareado excluindo zero **e**
Diebold-Mariano p<0,05 em ao menos 3 de 6 horizontes), o 6,09 **não bate** o naive sazonal
de 6,27: o intervalo é [-0,45, +0,10] e o DM dá 0 de 6. Nenhuma das doze combinações passa.

Análise em `scripts/analisa_variantes.py`, resultado em
`results/revisao/variants_vs_snaive.json`.

O padrão do que ajuda vale mais que o total: estatística de janela **piora** os dois
modelos, e o ganho vem da diferença sazonal. Ou seja, o que faltava aos boosters era
representação do ciclo anual, não capacidade.

## Adaptações feitas ao versionar

Marcadas no código com `# ADAPTADO`:

1. `REPO` apontava para uma cópia do repositório chamada `repo/` ao lado dos scripts, que
   era o layout do Drive. Agora aponta para a raiz deste repositório.
2. `run_optuna.py` importava `calendar_exog` de `variants.py` no topo do módulo.
   `variants.py` ainda não foi versionado, e o import no topo derrubaria também o modo
   `base`, que não usa essa função. O import passou para dentro de `forecast()`, no ramo que
   realmente precisa dele.

Fora isso, o código é o dele, sem alteração.

## O que ainda falta trazer do Drive

`run_covid.py`, `run_prophet_exog.py`, `run_temp_melhorado.py`, `check_dm_variance.py` e
`build_revisao_assets.py`.
