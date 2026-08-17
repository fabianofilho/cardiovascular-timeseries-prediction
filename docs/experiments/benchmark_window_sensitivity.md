# Sensibilidade ao Tamanho da Janela de Treino (expanding vs deslizante de 60 meses)

Data de execução: 2026-08-17

## Pergunta

Catorze anos de série (2010–2023) formam uma faixa temporal extensa: crescimento e envelhecimento populacional, mudanças de codificação e cobertura do SIM, e o choque da COVID tornam o processo não estacionário. O dado de 2010–2014 ajuda ou atrapalha a previsão de 2015–2023? A queda do sMAPE agregado de 7,3 para 4,8 entre rodadas não responde isso, porque o período de teste mudou junto; este experimento responde com pareamento.

## Desenho

- Mesmas 103 origens de teste do benchmark principal (min_train=60, horizonte 6, série de 168 meses).
- **Expanding**: treino do início da série até a origem (60 a 162 meses de treino).
- **Deslizante 60** (`--max-train-size 60` em `run_benchmark.py`): treino limitado aos últimos 60 meses antes da origem.
- Única diferença entre execuções é a quantidade de passado no treino. Bootstrap pareado por janela (B=10.000, seed 20260817) e Diebold-Mariano por horizonte, mesmos critérios pré-declarados do benchmark principal.

## Resultados

Arquivos: `results/benchmark_slide60_2010_2023_{metrics,predictions}.csv`, `results/uncertainty_window_sensitivity_*.csv`.

| model | sMAPE expanding | sMAPE deslizante 60 | diferença (pp) | IC95% da diferença | p_bootstrap | DM < 0,05 | veredito |
|---|---:|---:|---:|---:|---:|---|---|
| sarima | 4.80 | 5.46 | −0.66 | [−1.008, −0.346] | 0.000 | h2, h3, h4 | histórico ajuda, **confirmado** |
| prophet | 4.70 | 5.39 | −0.69 | [−1.159, −0.219] | 0.005 | nenhum | histórico ajuda, sugestivo |
| timesfm | 4.83 | 4.74 | +0.09 | [−0.125, 0.307] | 0.424 | nenhum | indiferente |
| xgboost | 6.94 | 6.87 | +0.07 | [−0.213, 0.354] | 0.635 | nenhum | indiferente |
| catboost | 6.59 | 6.64 | −0.05 | [−0.288, 0.176] | 0.661 | nenhum | indiferente |

(Diferença negativa = expanding melhor.)

## Leitura

1. **O passado além de 5 anos ajuda os modelos que o usam.** SARIMA perde 0,66 pp ao descartar o histórico anterior a 5 anos (IC exclui zero, DM significativo em 3 de 6 horizontes: melhoria confirmada pelos critérios pré-declarados). Prophet perde 0,69 pp (IC exclui zero, DM sem significância: sugestivo). O custo da não estacionariedade de 14 anos é menor que o ganho de sinal sazonal e de tendência nesta série.
2. **TimesFM é indiferente ao comprimento do histórico local** (diferença 0,09 pp, p=0,42). Consistente com o desenho do modelo: contexto limitado a 512 pontos com normalização interna, e o conhecimento vem do pré-treino, não da série alvo.
3. **Consequência para o regime de série curta:** na janela deslizante de 60 meses, o TimesFM (4,74) fica à frente de SARIMA (5,46) e Prophet (5,39), com DM abaixo de 0,05 contra o SARIMA em h3, h4 e h5. Com pouco histórico local, o foundation model tem vantagem; com histórico longo, os clássicos alcançam e o empate se restabelece. Esta é a caracterização mais informativa do papel do TimesFM medida até aqui.
4. A comparação entre rodadas continua sendo o recorte de mesmas datas (ganho de 1,2 a 1,7 pp), não o sMAPE agregado (7,3 vs 4,8), que mistura períodos de dificuldade distinta.

## Limite

A série é de contagens, não de taxas; parte da tendência absorvida pelo histórico longo é demografia (população e envelhecimento). Modelar taxa por população residente removeria essa componente e é candidato a próxima iteração, junto com a estratificação etária (Prioridade 4 do TODO).
