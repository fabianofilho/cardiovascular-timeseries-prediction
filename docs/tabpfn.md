# TabPFN: o que está verificado e o que não está

Autoria da rodada: **Isabella Saade** (isasaade23@gmail.com), `tabpfn_benchmark.ipynb`,
setembro de 2026. Resultado versionado em `results/revisao/tabpfn_resultados_v4.json`.

## Por que existe

O parecer pediu, no comentário 01:

> Entendo que não é indispensável e que o TimesFM já cobre um pouco disso, mas trazer o
> resultado do tabpfn não agregaria? Já que tem XGBoost e catboost tbm. Afinal, o melhor do
> tabular precisa entrar pra ser justa a comparação.

É um pedido legítimo: o artigo compara duas famílias de boosting e conclui contra elas sem
incluir o modelo tabular que hoje é estado da arte.

## O que está verificado

A rodada usou o mesmo protocolo do benchmark, 103 janelas, horizonte 6, treino mínimo 60,
janela expansiva, com `tabpfn_client 0.5.3` e `thinking_mode=False`.

O JSON traz, junto com o TabPFN, a reprodução dos modelos já publicados, e ela serve de
controle da bancada dela:

| modelo | obtido | artigo | veredito |
|---|---:|---:|---|
| naive | 10,741134 | 10,741134 | bate |
| snaive | 6,269313 | 6,269313 | bate |
| snaive_drift | 6,177330 | 6,177330 | bate |
| catboost | 6,589326 | 6,589326 | bate |
| sarima | 4,801258 | 4,795652 | desvia 0,0056 |
| prophet | 4,704948 | 4,700965 | desvia 0,0040 |
| xgboost | 6,803473 | 6,935027 | desvia 0,1316 |
| **tabpfn** | **6,035091** | — | novo |

As três referências ingênuas e o CatBoost batem exato, o que valida a bancada. Os desvios
de SARIMA, Prophet e XGBoost são os três casos já documentados em
`xgboost_reprodutibilidade.md`, e o do XGBoost confirma o achado: o ambiente dela usa
xgboost 3.4.1, que é uma sétima versão medida.

## O que NÃO está verificado

**A afirmação de que o TabPFN bate as referências ingênuas.** Ela é verdadeira como
estimativa pontual: 6,035 contra 6,177 do naive sazonal com drift, uma vantagem de 0,14 pp.

Mas o paper não decide por estimativa pontual. O critério pré-declarado exige intervalo de
bootstrap pareado excluindo zero **e** Diebold-Mariano com p<0,05 em ao menos 3 de 6
horizontes. Esse teste **não foi feito**, e não pode ser feito com o que está disponível: o
JSON guarda só o sMAPE agregado, e o critério precisa das previsões janela a janela.

Isso não é formalidade. A variante `catboost_direct` chegou a 6,092, uma vantagem de
**0,177 pp** sobre o naive sazonal, maior que a do TabPFN, e mesmo assim reprovou: intervalo
[-0,45, +0,10] contendo zero e Diebold-Mariano 0 de 6. Uma vantagem menor que essa,
avaliada nas mesmas 103 janelas sobrepostas, tem pouca chance de passar.

Ou seja, o que se pode afirmar hoje é que **o TabPFN é o melhor modelo tabular testado**,
à frente do CatBoost por 0,554 pp. O que **não** se pode afirmar é que ele bate uma regra
sem modelo.

## O que falta para fechar

Uma coisa só: o CSV de previsões por janela e horizonte da rodada do TabPFN, no mesmo
formato dos demais (`model, window, horizon, date, y_true, y_pred`). Com ele,
`scripts/analisa_variantes.py` roda o teste em minutos e a afirmação passa a ter o mesmo
suporte que todas as outras do paper.

Enquanto isso, o manuscrito reporta o número e declara que o teste pareado está pendente,
em vez de afirmar a vitória.

## O que não entrou

`timesfm` e `tabpfn_ts` não foram rodados nessa sessão. O TimesFM já tem valor no artigo e
pede GPU. O TabPFN-TS é o enquadramento de série temporal em vez de tabular, e não é o que
o parecer pediu: o pedido era pelo competidor tabular do XGBoost e do CatBoost.
