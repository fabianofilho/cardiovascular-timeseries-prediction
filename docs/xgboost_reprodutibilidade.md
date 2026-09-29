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
| xgboost 3.4.1 (ambiente da Isa, Colab) | 6.803473 |
| **valor guardado em `results/`** | **6.935000** |

A 3.4.1 vem da rodada independente da Isabella Saade no Colab, com outra máquina e outro
sistema operacional, e é o sétimo valor distinto. Ela reproduz exato o CatBoost, o
`snaive`, o `snaive_drift` e o `naive`, o que isola o comportamento no XGBoost.

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

## Onde isso já apareceu no texto

A tabela de ajuste de hiperparâmetros nasceu com cada coluna medida no ambiente de quem
rodou a busca. Como a base do XGBoost lá era 6,85 (2.x) e aqui é 6,83 (3.2.0), o mesmo ganho
saía como 0,21 pp na tabela e 0,23 pp na prosa. Não era erro de conta, era mistura de
ambientes, e foi a Isa quem localizou.

Resolvido reavaliando os hiperparâmetros vencedores aqui, em
`scripts/reproduz_optuna_fixado.py`. O CatBoost reproduz os três valores de origem com
divergência **0,0000**; o XGBoost reproduz o modo honesto exato (7,064875) e diverge na base
(0,0211) e no teto (0,0253). É o mesmo comportamento desta página, e é por isso que a tabela
passou a usar o medido no ambiente fixado.

| coluna | CatBoost | XGBoost |
|---|---:|---:|
| sem ajuste | 6,589326 | 6,832390 |
| busca honesta | 6,416290 | 7,064875 |
| teto com vazamento | 6,347102 | 6,249681 |

Efeito colateral que valia saber: o teto vazado do XGBoost, 6,2497, **passa** o naive sazonal
de 6,2693 por 0,02 pp. O texto antes dizia que nem o teto alcançava a referência, o que valia
para o CatBoost e não para o XGBoost. Agora declara a exceção.

## Decisão tomada, 2026-09-23

**Regenerado.** O valor publicado passa de 6,9350 para **6,880428**, medido com xgboost
3.2.0 e `n_jobs=1`.

Fixar a versão não bastava: a mesma 3.2.0 dá 6,827 em 4 threads e 6,880 em 1. `n_jobs=1` é
a única escolha que não depende da máquina, inclusive num runner de CI com um núcleo. Custa
tempo de ajuste e compra um número que outra pessoa consegue obter. Trocado em
`src/cv_timeseries/models.py`, em `scripts/revisao/variants.py`, em `run_optuna.py` e no
gerador do notebook do TabPFN, para não sobrar um canto no ambiente antigo.

Antes de aplicar, `scripts/regen_xgboost.py` confirmou que com `n_jobs` fixo duas execuções
dão previsão a previsão idêntica.

### O que foi regenerado

Quatro rodadas, porque regenerar só a Tabela 1 deixaria a coluna "sem temperatura" da
Tabela 5 num ambiente e a "com temperatura" noutro:

| arquivo | rótulo | antes | depois | delta |
|---|---|---:|---:|---:|
| benchmark_sim_real_sp_2010_2023 | xgboost | 6,9350 | 6,8804 | -0,0546 |
| benchmark_exog_temp_climatology | xgboost_temp | 6,5528 | 6,5121 | -0,0407 |
| benchmark_exog_temp_observed | xgboost_temp | 6,5500 | 6,5250 | -0,0250 |
| benchmark_slide60_2010_2023 | xgboost_slide60 | 6,8657 | 6,8569 | -0,0088 |

Mais os três conjuntos de incerteza (semente 20260817), as doze variantes, a tabela do
Optuna e todos os assets do paper. Só as linhas do XGBoost foram substituídas; as dos
outros modelos ficaram byte a byte, porque prophet, sarima e catboost reproduzem com
divergência 0,0000 e o TimesFM não é reexecutável aqui.

### O que mudou de conclusão

Uma coisa. Na Tabela 5, o ganho da temperatura no XGBoost passou de **DM 2/6 para 3/6**, e
com o intervalo já excluindo zero ele passa a atender o critério pré-declarado. É a única
linha das quatro que atende, e está escrita no manuscrito.

O resto não se move: os três pares do top-3 continuam indistinguíveis com DM 0/6, a
comparação contra o naive sazonal continua não estabelecida, e o ranking é o mesmo.

Um efeito colateral vale nota: por horizonte a mudança chega a 0,247 pp (h=4), bem mais que
os 0,055 do agregado, porque os sinais se cancelam. É o que a fig3 desenha.

### Decisão anterior, mantida como registro

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

**Resolvido em 2026-09-23.** A causa era a versão do statsmodels, não o otimizador. No
`.venv` criado a partir dos requirements, com **statsmodels 0.15.0**, o SARIMA reproduz o
guardado com divergência **0,0000**; a divergência de 0,013 pp aparecia no Python do sistema,
que tinha 0.14.6. As duas janelas que caíam em ótimo local diferente eram efeito de versão,
o mesmo mecanismo do XGBoost porém menor e agora fechado. A versão exata está em
`requirements-lock.txt`.

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

---

# Rodada de 2026-09-28: o que muda de máquina para máquina

Medido num Mac (arm64, Python 3.12) com as versões exatas do lock para os pacotes
envolvidos: xgboost 3.2.0, catboost 1.2.10, prophet 1.4.0, numpy 2.5.3, pandas 2.3.3,
scikit-learn 1.9.1, skforecast 0.25.0. A máquina que produziu os números publicados em
23/09 não é esta.

## XGBoost: `n_jobs=1` não basta entre plataformas

A decisão de 23/09 dizia que `n_jobs=1` é a única escolha que não depende da máquina. Não
se sustenta entre plataformas:

| rodada | publicado (23/09) | este Mac, `n_jobs=1` |
|---|---:|---:|
| xgboost, sem temperatura | 6,880428 | 6,833274 |
| xgboost_temp, climatologia | 6,512106 | 6,410349 |
| ganho da temperatura | 0,368 | 0,423 |

`n_jobs=1` torna o resultado estável **dentro** de uma plataforma. Entre plataformas, a
aritmética de ponto flutuante do histograma ainda difere. O número publicado continua
sendo de um ambiente específico, agora identificado por versão e thread, mas não por CPU.

## O veredito da temperatura no XGBoost está no limiar

O critério pede intervalo pareado excluindo zero e Diebold-Mariano com p<0,05 em ao menos 3
de 6 horizontes. Nos três ambientes medidos:

| ambiente | ganho | IC 95% | DM p<0,05 | atende |
|---|---:|---|---:|---|
| anterior a 23/09 | 0,327 | exclui zero | 2 de 6 | não |
| publicado (23/09) | 0,368 | [0,190, 0,548] | 3 de 6 | sim |
| este Mac | 0,423 | [0,268, 0,574] | 4 de 6 | sim |

A única linha da Tabela 5 que atende o critério atende em dois de três ambientes. O
manuscrito afirma que atende, e isso é verdade no ambiente publicado, mas a conclusão
depende da máquina. Decisão pendente do autor: manter como está, ou declarar essa
sensibilidade no texto.

## A tabela do termo de calendário ficou para trás na regeneração

`results/revisao/temp_melhorado.csv` e `paper/tables/revisao_tab_temp_diffcal.tex` ainda
trazem o XGBoost de antes de 23/09 (6,832 → 6,505, ganho 0,327), enquanto a Tabela 5 e o
resumo usam 6,880 → 6,512, ganho 0,368. A regeneração de 23/09 não incluiu
`scripts/revisao/run_temp_melhorado.py`.

Rodado neste Mac, o script dá XGBoost base 6,833 → 6,410 (+0,423) e diffcal 6,285 → 6,454
(-0,169). O controle falha (a base deveria dar 6,8804), então esse resultado **não** foi
versionado. O mecanismo se mantém nos dois ambientes medidos: com o termo de calendário, o
ganho da temperatura no XGBoost desaparece ou inverte de sinal (-0,036 antes, -0,169 aqui).
Para corrigir a tabela, rodar o script na máquina que gerou os números de 23/09.

## Prophet: semente resolve a banda, mas o ponto varia entre máquinas

`scripts/revisao/check_prophet_semente.py`, resultado em `results/revisao/prophet_semente.json`:

- Com a semente do `run_calibracao.py`, dois ajustes da mesma janela dão ponto e banda
  idênticos (diferença 0,00). Sem semente, a banda move 34 a 53 óbitos. A nota da Tabela 6,
  que dizia o contrário, foi corrigida e agora cita esses números a partir do JSON.
- Entre máquinas, a previsão **pontual** muda em 63 das 103 janelas, até 49 óbitos. O PICP
  fica idêntico (0,772) e o sMAPE vai de 4,701 para 4,700. A afirmação do apêndice acima,
  "ponto determinístico, divergência 9,09e-13", vale dentro da máquina original.
