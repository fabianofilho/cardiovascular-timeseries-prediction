# O que entra, o que sai e o que continua aberto

Situação em 23/09/2026, depois da regeneração do XGBoost (`792fe6c..e829dc2`), da
conferência em ambiente limpo e da medição do modelo tabular nas 103 janelas.

Três fontes cruzadas: o resumo da regeneração, o parecer de cinco comentários
(`manuscript_comentado.pdf`) com o cross-check de 18/08 que o acompanha, e o estado atual
de `paper/manuscript.tex`.

---

## 1. Delta contra o resumo da regeneração

| item | estado |
|---|---|
| regeneração do XGBoost em quatro rodadas | **confere.** Regerar os assets hoje não move mais nenhuma tabela nem figura: o que está versionado é o que a fonte produz |
| Tabela 5, temperatura passa de DM 2/6 para 3/6 | **confere**, e continua sendo a única das quatro linhas que atende o critério |
| linha do Prophet duplicada na Tabela 5 | resolvida; a implementação que ficou passa pelo mesmo bootstrap das demais |
| teste que fixava 6,9350 no código | resolvido, e invertido: agora cobra o valor da configuração fixada |
| checar se a regeneração trocou outros símbolos | **sim, trocou, e mais do que se supunha** — ver §4 |
| CSV de previsões por janela do modelo tabular | **fechado** — ver §5 |
| Limitações afirmando que não houve busca de hiperparâmetros | **resolvido pela fusão.** O manuscrito atual reporta o estudo de ajuste e traz a tabela correspondente |
| qual rodada de calibração é canônica | **não determinado, e a premissa precisa ser revista** — ver §6 |

Dois achados que não estavam no resumo:

**As colunas de IC e DM da Tabela 7 não são do XGBoost.** Elas ficaram idênticas depois da
regeneração porque são do par CatBoost contra naive sazonal, como o cabeçalho e a legenda
já dizem, e o gerador confirma (`d = vj["modelos"][f"catboost_{v}"]`). O IC medido do
`catboost_base`, `[-0,06, +0,72]`, bate com o da tabela. Não havia número velho.

**O commit da regeneração quebrou o gerador do notebook do modelo tabular.** Dois `n_jobs`
foram trocados com um comentário no fim da linha, e o comentário engoliu o que fechava a
expressão: numa a vírgula final, inofensiva, na outra o parêntese do `dict` do espaço de
busca. O gerador passou a falhar na validação. Corrigido.

---

## 2. Os cinco comentários do parecer

| # | comentário | estado hoje |
|---|---|---|
| C1 | falta o objetivo ao fim do Background | **atendido pela fusão.** A frase de objetivo entrou no resumo |
| C2 | incluir o modelo tabular de referência | **atendido**, com o teste pareado que faltava — ver §5 |
| C3 | frase do critério de decisão solta e incompleta | atendido: o critério está declarado em Métodos, com as duas condições e o que acontece quando só uma é satisfeita |
| C4 | enriquecer a engenharia de atributos | atendido: doze variantes, tabela própria, e a conclusão de que nenhuma resgata os modelos de boosting |
| C5 | como foi o ajuste de hiperparâmetros | atendido: estudo de ajuste com tabela, incluindo o teto com vazamento declarado como teto |

## 3. Os achados R1–R9 do cross-check

| # | achado | estado hoje |
|---|---|---|
| R1 | não havia baseline ingênua nem MASE | atendido: as três referências ingênuas entraram, MASE é reportado e justificado |
| R2a | "nenhum destes modelos produz intervalo" era falso | atendido: a afirmação saiu e há seção própria sobre calibração |
| R2b | Prophet fora da comparação com exógena por motivo que não procede | atendido: entrou via `add_regressor()` e é a linha que mostra o ganho inverso à sazonalidade já embutida |
| R3a | métrica e perda inconsistentes | declarado em Métodos |
| R3b | DM em h=1 infla significância | **atendido pela fusão** — ver §4 |
| R3c | bootstrap não trata dependência entre janelas | **atendido pela fusão**: a limitação entrou nomeada, com o bootstrap de blocos móveis apontado como a alternativa mais estrita |
| R3d | sem ajuste de multiplicidade em ~100 testes | **atendido pela fusão**: declarado, com o critério pré-declarado posto no lugar do ajuste formal e explicitamente não confundido com controle de taxa de erro |
| R3e | COVID sem análise de sensibilidade | atendido: seção própria |
| R4 | reprodutibilidade | atendido: ambiente fixado, 108 versões exatas |
| R5 | cinco pendências que travam a submissão | **quatro resolvidas** — ver §7 |
| R6 | intervalos não calibrados, o ranking se inverte | atendido: seção própria e tabela |
| R7 | sensibilidade ao período COVID | atendido |
| R8 | a especificação corrigida muda as Tabelas 4 e 5 | absorvido nas tabelas regeradas |
| R9 | a preocupação sobre o DM em h=1 não se confirma | **atendido pela fusão** — ver §4 |

---

## 4. Símbolos de LaTeX, e o que a checagem achou

A regeneração trocou símbolos, e em mais lugares do que a suspeita inicial. O inventário
em modo matemático nas duas árvores mostra **dois** comandos entrando, não um: além do
`\blacksquare` de `fig4_janela.tex`, entraram `\blacksquare` em
`revisao_fig_enriched_janela.tex` e `\square` em `revisao_fig_temp_diffcal.tex`. São
**três arquivos** dependentes de `amssymb`, e o comentário do preâmbulo citava um.

Nada está quebrado, porque o pacote está carregado. Os três símbolos saem dos geradores,
então sobrevivem a qualquer regeneração. O `\blacksquare` da fig4 acompanha uma troca real
de marcador, de círculo para quadrado: a legenda seguiu o desenho, não é símbolo trocado à
toa.

Dois testes novos passam a cobrar a correspondência entre símbolo e pacote, e a cobrar que
símbolo novo seja catalogado antes de entrar. Sem essa segunda metade, a próxima troca
passaria igual a esta.

A checagem de robustez do DM entrou junto, vinda do manuscrito paralelo: recalcular todas
as células com kernel de Bartlett e banda de Newey-West, que absorve a dependência entre
janelas sobrepostas, deixa as 90 do mesmo lado do limiar de 5%. É a resposta medida a R3b
e R9, e estava ausente do texto principal.

---

## 5. O modelo tabular, medido

103 janelas completas, **sMAPE 5,8459**, o melhor modelo tabular do trabalho.

| contra | Δ (pp) | IC 95% | DM | critério |
|---|---:|---|---:|---|
| naive sazonal | −0,4234 | [−0,698, −0,128] | 1/6 | não atende |
| naive sazonal com drift | −0,3315 | [−0,622, −0,031] | 1/6 | não atende |
| naive | −4,8953 | [−6,059, −3,722] | 6/6 | atende |
| melhor variante de boosting | −0,2463 | [−0,557, +0,071] | 1/6 | não atende |
| Prophet | +1,1449 | [+0,730, +1,593] | 2/6 | não atende |
| SARIMA | +1,0502 | [+0,637, +1,483] | 4/6 | não atende |

**Contrariou a previsão registrada antes da medição, nas duas direções.** Esperava-se
intervalo contendo zero, como o da melhor variante de boosting; o intervalo excluiu zero, e
é o único modelo tabular do trabalho de que isso vale. O que reprova é a outra metade do
critério.

O DM por horizonte diz mais que o agregado: só h=1 é significativo (p=0,0012) e a vantagem
some dali em diante, com p acima de 0,35 a partir de h=3. A vantagem sobre a regra sazonal
é de um mês à frente, e não sobrevive ao horizonte de seis meses que é o do trabalho.

**Ressalva que precisa estar no texto onde o número aparece.** A rodada de setembro deu
6,035 com o cliente 0.5.3; esta deu 5,846 com o 0.6.0. São 0,19 pp, mais que os 0,13 pp do
XGBoost que motivaram fixar o ambiente inteiro. E é pior que o caso do XGBoost: o modelo
roda num servidor, a versão não entra no arquivo de travas e pode mudar sem aviso. Esta é a
única linha do trabalho que não é reproduzível por quem clonar o repositório.

---

## 6. A calibração, e por que não fecha pelo caminho proposto

A proposta era fechar regenerando a figura a partir do script, "que já semeia". Ele semeia
mesmo, por janela, com justificativa registrada na declaração da semente, de 19/08: a
previsão pontual é determinista, mas a banda vem de amostragem posterior, e sem fixar isso
a cobertura muda a cada execução.

O problema é que o repositório afirma as duas coisas. A nota da Tabela 6, escrita **três
semanas depois**, diz que a amostragem não é semeada e que rodar de novo move os limites em
até cem óbitos. Quem mediu em setembro mediu com a semente já no lugar.

As duas medições foram feitas sem versão fixada da biblioteca, que só passou a ser fixada
em 23/09. As duas podem estar certas, em versões diferentes.

Com a versão agora travada, a pergunta é uma só e custa segundos: dois ajustes da mesma
janela, com a mesma semente antes de cada um, comparando ponto e banda. Ponto idêntico com
banda idêntica e o item fecha pelo caminho proposto, e a nota da Tabela 6 sai do
manuscrito. Ponto idêntico com banda diferente e regenerar a figura só congela uma amostra,
e a escolha passa a ser entre semear o gerador certo ou declarar a rodada versionada como
referência e parar de chamá-la de semeada.

A medição está pronta no notebook de conferência e não usa API. **Não responder este item
antes de rodá-la**, porque o desfecho decide entre duas ações opostas.

---

## 7. O que sai do texto, e o que fica aberto

**Saiu:**

- a afirmação de que o teste pareado do modelo tabular estava pendente, e os números de
  sMAPE 6,04 associados a ela
- quatro das cinco marcas de pendência, preenchidas com texto que já existia no manuscrito
  paralelo: contribuições CRediT, ética com a Resolução 510/2016, e o objetivo ao fim do
  Background; a de financiamento virou escolha entre duas alternativas redigidas

**Fica aberto, e é pouco:**

1. **Financiamento.** Duas alternativas redigidas no texto, uma com aporte e outra sem. É
   decisão factual, não de redação.
2. **Completude do registro de óbitos.** O manuscrito paralelo cita 99,9% para 2015–2016
   com uma referência que **a própria entrada bibliográfica declara não verificada na
   fonte**. Por isso a afirmação não foi trazida: a marca continua, agora apontando qual
   referência conferir.
3. **Agradecimentos.** Sem par no outro manuscrito.
4. **Ordem do SARIMA fixada sem diagnóstico de resíduos.** Fixar é defensável, mas precisa
   ser declarado como escolha. Não está.
5. **Data no cover letter.**
6. **`fillna(0.0)` após o resample** em `data.py`: mês faltante viraria zero em silêncio. O
   validador cobre o caso hoje, mas a linha continua lá.
7. **A rodada anterior de 60 meses** citada na Tabela 3 é auto-citação de trabalho não
   publicado. Sem preprint, DOI ou etiqueta de versão, não é verificável por quem avalia.

---

## 8. Procedência do que foi juntado

O texto base é o do repositório, que tem a estrutura nova: as quatro subseções que a
revisão pediu só existem nele. Do manuscrito paralelo atravessaram sete blocos, todos
ausentes do base e nenhum deles número: objetivo no resumo, contribuições CRediT, ética,
financiamento, checagem de Bartlett e Newey-West, dependência entre janelas, e
multiplicidade. Duas referências canônicas entraram junto, com DOI.

Nenhuma tabela e nenhuma figura mudou com a fusão, o que é o esperado: é texto, não
medição. A suíte de testes passa inteira.
