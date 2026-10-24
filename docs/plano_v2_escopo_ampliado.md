# Plano da v2: muitas séries curtas em vez de uma série longa

Estado em 24/10/2026. A v1 (uma série estadual de SP, 168 meses, cinco modelos) levou desk
reject do IJF no dia da submissão (IJF-D-26-01619): texto com marcas de agente e
contribuição insuficiente "in view of what can be achieved with agentic AI". A v1 fica
congelada na tag `ijf-v1-2026-10-24`. A v2 muda a pergunta, não só a prosa. Regras de
escrita em `~/.claude/skills/paper/references/contribuicao-e-voz.md`.

## A pergunta da v2

A própria Discussion da v1 disse onde o modelo de fundação vale: vigilância com muitos
indicadores, unidades ou estratos, onde nenhuma série é longa. A v1 não testou isso. A
v2 testa:

> Na vigilância de mortalidade cardiovascular das 27 UFs, em quais séries (por tamanho da
> população, comprimento do histórico e subgrupo de causa) um modelo zero-shot entrega
> previsão utilizável e intervalo calibrado onde os modelos clássicos não entregam?

Produto para o leitor: um mapa de onde cada abordagem serve, com a fronteira em número de
óbitos por mês e meses de histórico, não um ranking.

## Barra de contribuição (resposta escrita antes de qualquer código)

O que a v2 tem que um agente não faz em um dia com os mesmos dados:

1. **Escopo que muda a conclusão.** Centenas de séries cobrindo o espaço de volume (RR
   tem dezenas de óbitos cardiovasculares por mês, SP tem sete mil) e de comprimento. A
   conclusão da v1 ("o comprimento da série importa mais que o modelo") vira uma curva
   com fronteira, testável.
2. **Decisões de domínio que mudam o desenho.** Taxa por população em vez de contagem
   (limitação da v1); subgrupos de causa com sazonalidade distinta (isquêmica I20-I25,
   cerebrovascular I60-I69, insuficiência cardíaca I50, hipertensiva I10-I15); completude
   do SIM, que varia entre regiões e muda o que é erro de modelo e o que é erro de
   registro; códigos garbage (R00-R99) como sensibilidade nas UFs com pior qualidade.
3. **Intervalo como desfecho principal.** Cobertura com conformal (o `sensibilidade_conformal.py`
   já existe, LAB-126) para todos os modelos, por estrato. Para o gestor o intervalo é a
   quantidade operativa; a v1 só mediu o problema.
4. **Diálogo com a literatura nomeada.** M4 e M5 (séries curtas e hierárquicas), Monash e
   GIFT-Eval (benchmarks de modelos de fundação), Chronos e TimesFM, e a literatura de
   previsão de mortalidade no Brasil. Ler e citar antes de escrever a Discussion.
5. **Voz de quem usa o dado.** Falta um coautor ou interlocutor da vigilância (SES ou
   Ministério) que diga qual horizonte e qual granularidade a decisão usa. Sem isso o
   item 1 continua sendo um benchmark maior. **Risco aberto**: decidir com quem falar.

O que não conta como contribuição e não será vendido como tal: mais modelos, Optuna,
código aberto, intervalos de confiança no ranking.

## Desenho

- **Séries:** 27 UFs x 4 subgrupos de causa x (total, sexo, 3 faixas etárias: <60, 60-79,
  80+). Na ordem de 650 séries mensais, 2010-2023. Séries com média abaixo de um piso
  (definir, por exemplo 5 óbitos por mês) entram só na análise de fronteira.
- **Alvo:** taxa por 100 mil, com denominador por UF, ano, sexo e faixa etária. Contagem
  como sensibilidade.
- **Modelos:** naive sazonal, ETS, SARIMA, TimesFM 2.5 zero-shot, Chronos-Bolt zero-shot.
  Prophet e boosting só se o custo for baixo: o XGBoost da v1 só reproduz no Windows.
- **Comprimento:** cada série avaliada com 24, 36, 60 e 120 meses de histórico nas mesmas
  datas de teste (o desenho pareado da v1, que funcionou).
- **Métricas:** MASE e sMAPE pontuais; cobertura e largura do intervalo conformal a 80 e
  95%; teste pareado por bloco como na v1.
- **Análise principal:** diferença de erro e de cobertura entre zero-shot e clássicos em
  função de volume e comprimento (modelo de efeitos mistos ou GAM sobre as séries).

## Dados a obter

| Item | Fonte | Situação |
|---|---|---|
| SIM 2010-2023, 27 UFs | FTP DataSUS via `scripts/extract_sim_real.py --uf` | só SP extraído; o lake do omnisus tem 27 UFs mas só 2018-2023 |
| População por UF, ano, sexo e idade | estimativas DataSUS/IBGE (POPSVS) | a obter; o lake só tem total municipal de 2010 e 2022 |
| Completude do SIM por UF | literatura (busca ativa, Szwarcwald e outros) | a levantar, com referência verificada |

## Passos

1. Taguear a v1 (`ijf-v1-2026-10-24`) e não mexer mais em `paper/` da v1.
2. Extrair SIM 2010-2023 das 26 UFs restantes, com a validação que a v1 já usa.
3. Obter os denominadores e montar o painel de séries (UF x causa x estrato).
4. Rodar o benchmark de comprimento com os cinco modelos e o conformal.
5. Ler a literatura do item 4 da barra e fechar a análise de fronteira.
6. Fabiano escreve o núcleo: pergunta, por que importa para a vigilância, achado
   principal, frase da Discussion que defende em voz alta. O manuscrito cresce em volta.
7. `/paper-tex` com a skill nova; `marcas_ia.py` sem falha; `/revisor` com a passada M;
   `/journal` para o alvo (IJF de novo só se a barra passar com folga).
8. Avisar Isa, Victor e Alexandre da rejeição e do plano antes do passo 2.

## Decisões em aberto

- Interlocutor da vigilância (item 5 da barra).
- Piso de volume para entrar na análise principal.
- Se a v2 mantém a temperatura (só faz sentido com resolução semanal, que o SIM permite).
