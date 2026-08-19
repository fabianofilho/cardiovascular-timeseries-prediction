# O XGBoost muda de resultado conforme o ambiente

Estado: aberto. Afeta uma linha da Tabela 1 do manuscrito.

## O que foi medido

Mesmo código, mesma série, mesmas 103 janelas, mesma semente. Só o ambiente muda.

| Ambiente | sMAPE do XGBoost base |
|---|---:|
| xgboost 2.0.3 | 6.854448 |
| xgboost 2.1.4 | 6.854448 |
| xgboost 3.0.2 | 6.832390 |
| xgboost 3.1.0 | 6.832390 |
| xgboost 3.2.0 | 6.832390 |
| xgboost 3.3.0 | 6.984887 |
| **valor guardado em `results/`** | **6.935000** |

Dentro de um mesmo ambiente o resultado é determinístico: três execuções seguidas dão
spread `0.00e+00`. A variação é entre ambientes, não entre execuções.

A contagem de threads move o resultado sozinha, com a versão fixa em 3.2.0:

| `n_jobs` | sMAPE |
|---|---:|
| 1 | 6.880428 |
| 2 | 6.861767 |
| 4 | 6.827159 |
| 8 | 6.834284 |
| default (12 núcleos) | 6.832390 |

Ou seja, fixar a versão não basta: a mesma versão em uma máquina com outro número de
núcleos dá outro número. A amplitude observada juntando versão e threads é de cerca de
0,15 pp.

O CatBoost não tem esse comportamento. Ele deu `6.589326` em todos os ambientes testados,
contra `6.5893` guardado em `results/`, e reproduz o valor publicado.

## O que isso explica

Os dois valores divergentes que apareceram na auditoria do trabalho do Victor Ovani
Marchetti não eram um erro dele:

- `baseline_reproduzida = 6.8535` corresponde ao xgboost 2.x (medimos 6.854448)
- `baseline_esperada = 6.8324` corresponde exatamente ao xgboost 3.0 a 3.2 (medimos 6.832390)

Ele tinha dois ambientes, e a "divergência" era a diferença entre eles. Confirmado ao rodar
os hiperparâmetros dele no nosso backtest sob xgboost 3.2.0: o CatBoost tunado reproduz
`6.4162903089` e o XGBoost tunado reproduz `7.0648751870`, ambos com divergência da ordem
de `1.7e-11`, que é ruído de ponto flutuante. O estudo de Optuna dele está correto.

## O que fica em aberto

O valor `6.9350` guardado em `results/` não é reproduzido por nenhuma das seis versões
testadas nem por nenhuma contagem de threads. O ambiente que o produziu não foi
identificado, e pode envolver plataforma (os resultados podem ter vindo de Linux) além de
versão.

Consequência: a linha do XGBoost na Tabela 1 do manuscrito não é reproduzível hoje.

## O que não muda

Nenhuma conclusão do paper depende do dígito. Em todos os ambientes testados, o intervalo
do XGBoost vai de 6.83 a 6.98, e em todos eles:

- é o pior dos cinco modelos (o CatBoost fica em 6.59)
- perde para o naive sazonal (6.27)
- fica muito atrás dos três líderes (4.70 a 4.83)

A comparação de tuning também é robusta: sob um mesmo ambiente (3.2.0), o XGBoost vai de
6.832390 para 7.064875 com os hiperparâmetros otimizados, ou seja, piora 0,23 pp. O sinal
não depende de qual ambiente se escolhe.

## Decisão pendente

Regenerar a linha do XGBoost com um ambiente fixado obrigaria a regenerar também as tabelas
de comparação pareada, os testes de Diebold-Mariano e as figuras que dependem dele. É uma
mudança de números publicados e não deve ser feita em silêncio. Está registrada aqui para
ser decidida, não aplicada por conta própria.

Enquanto isso, `requirements-optional.txt` fixa a versão, para que ao menos execuções
futuras concordem entre si.
