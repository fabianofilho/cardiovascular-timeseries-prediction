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

---

# Apêndice: o que a rodada de calibração encontrou nos outros dois modelos

Ao reproduzir os intervalos de previsão (`scripts/run_calibracao.py`), a checagem de
procedência comparou a previsão pontual gerada ali contra a guardada em `results/`. O
resultado separa os três modelos em três comportamentos diferentes.

## Prophet: ponto determinístico, banda estocástica

A previsão pontual reproduz o benchmark com divergência `9.09e-13`, ou seja, é determinística.

A **banda**, não. O Prophet gera o intervalo por amostragem posterior e não semeia esse
sorteio. Três ajustes na mesma janela, com os mesmos dados:

| execução | limite inferior (h=1) | limite superior (h=1) | largura média |
|---|---:|---:|---:|
| 1 | 6111,08 | 6895,20 | 781,3 |
| 2 | 6107,68 | 6883,35 | 764,4 |
| 3 | 6087,41 | 6858,28 | 777,1 |

Divergência máxima entre execuções: 31 óbitos no limite inferior e 48 no superior. Isso move
o PICP em cerca de meio ponto percentual por execução, e explica a diferença entre os 77,0%
reportados pelo Victor e os 77,5% da primeira reprodução aqui.

`np.random.seed()` antes do ajuste resolve: com semente fixa a divergência cai para
`0.00e+00`. O script agora semeia a cada janela, e não uma vez no início, para que rodar um
subconjunto das janelas dê o mesmo número.

## SARIMA: quase reprodutível, com duas janelas fora

A previsão pontual diverge da guardada, mas o perfil é muito diferente do XGBoost:

| medida | valor |
|---|---:|
| divergência mediana | 0,0044 óbitos |
| divergência média | 1,1464 óbitos |
| divergência máxima | 127,8208 óbitos |
| janelas com divergência acima de 1 óbito | 2 de 103 |
| sMAPE guardado | 4,795652 |
| sMAPE agora | 4,808559 |

Ou seja, 101 das 103 janelas são idênticas até a quarta casa. Em duas janelas o otimizador
de máxima verossimilhança converge para um ótimo local diferente, o que é coerente com os
avisos de `Non-invertible starting seasonal moving average` emitidos durante o ajuste.

Confirmado que a causa não é o script novo: a própria classe `SarimaForecaster` do
repositório, rodada sem alteração, produz a mesma divergência.

Efeito sobre o paper: o sMAPE do SARIMA passa de 4,7957 para 4,8086, ou seja, muda de 4,80
para 4,81 no arredondamento da Tabela 1. Menor que a diferença entre XGBoost e o valor
guardado, e sem efeito sobre a calibração, porque 101 das 103 janelas são as mesmas.

## Resumo dos três

| Modelo | Ponto reproduz? | Causa quando não |
|---|---|---|
| Prophet | sim, `9e-13` | banda precisava de semente, já corrigido |
| SARIMA | quase, 2 janelas de 103 | ótimo local do otimizador de verossimilhança |
| XGBoost | não | versão da biblioteca e contagem de threads |

Só o XGBoost tem divergência estrutural. Os outros dois são casos localizados, e o do
Prophet já está fechado.
