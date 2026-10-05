# TabPFN: o que está verificado e o que não está

Rodada de setembro de 2026, por `tabpfn_benchmark.ipynb`. Resultado versionado em
`results/revisao/tabpfn_resultados_v4.json`.

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

## O teste pareado, feito

Rodada de 23/09/2026, 103 janelas completas, com `tabpfn_client 0.6.0`. O CSV por janela
está em `results/revisao/tabpfn_predictions.csv` e o teste é o mesmo que julga todas as
outras afirmações do trabalho: bootstrap pareado com 10.000 reamostragens sobre as mesmas
janelas, semente 20260817, e Diebold-Mariano com correção de Harvey, Leybourne e Newbold.

**sMAPE 5,8459.**

| contra | Δ (pp) | IC 95% | DM | critério |
|---|---:|---|---:|---|
| naive sazonal | −0,4234 | [−0,698, −0,128] | 1/6 | não atende |
| naive sazonal com drift | −0,3315 | [−0,622, −0,031] | 1/6 | não atende |
| naive | −4,8953 | [−6,059, −3,722] | 6/6 | atende |
| `catboost_direct` | −0,2463 | [−0,557, +0,071] | 1/6 | não atende |
| Prophet | +1,1449 | [+0,730, +1,593] | 2/6 | não atende |
| SARIMA | +1,0502 | [+0,637, +1,483] | 4/6 | não atende |

O critério pré-declarado exige as duas coisas: intervalo excluindo zero **e** DM com p<0,05
em ao menos 3 de 6 horizontes. Contra as duas referências sazonais o TabPFN cumpre a
primeira e falha a segunda, então **não atende**. Contra o naive simples atende, o que não
diz muita coisa: qualquer modelo do trabalho atende.

O padrão do DM explica o resultado melhor que o agregado. Só o primeiro horizonte é
significativo, e a vantagem some conforme o horizonte cresce:

| h | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| p | **0,0012** | 0,0933 | 0,3586 | 0,7067 | 0,6752 | 0,7440 |

A vantagem do TabPFN sobre a regra sazonal é de curto prazo, e um mês à frente. Num
horizonte de seis meses, que é o do trabalho, ela não se sustenta.

O resultado contrariou a previsão registrada antes da medição, e nas duas direções. A
expectativa era de intervalo contendo zero, como aconteceu com o `catboost_direct`; o
intervalo excluiu zero, o que é mais forte que o de qualquer variante de boosting testada.
E o sMAPE veio 5,846, não os 6,035 da rodada de setembro.

**Ressalva de reprodutibilidade, e ela é séria.** Os 6,035 de setembro saíram do
`tabpfn_client 0.5.3`; os 5,846 daqui, do 0.6.0. São 0,19 pp de diferença, mais que os
0,13 pp do XGBoost que motivaram fixar o ambiente inteiro. E o caso do TabPFN é pior: o
modelo roda num servidor, então a versão dele não entra no `requirements-lock.txt` e pode
mudar sem aviso. Diferente de todas as outras linhas do trabalho, esta não é reproduzível
por quem clonar o repositório — é uma medição datada. O texto precisa dizer isso onde
reportar o número.

## O que se pode afirmar

Que **o TabPFN é o melhor modelo tabular testado**, à frente do CatBoost por 0,74 pp e do
melhor `catboost_direct` por 0,25 pp, e que é o único deles cujo intervalo pareado contra a
regra sazonal exclui zero.

O que **não** se pode afirmar é que ele bate uma regra sem modelo pelo critério do trabalho.
E continua valendo a conclusão do artigo: SARIMA e Prophet ficam à frente dele, e contra o
SARIMA a diferença tem DM 4 de 6, ou seja, está estabelecida.

## Como reproduzir a rodada

O CSV por janela está versionado em `results/revisao/tabpfn_predictions.csv`, e o teste
sai dele em minutos com `PYTHONPATH=src python scripts/analisa_variantes.py`, que o
carrega junto com os demais sem alterar nenhum dos outros modelos.

Para medir de novo, do zero, pelo mesmo caminho que gerou todas as outras linhas do
artigo:

```
python scripts/run_benchmark.py \
    --input-csv results/series/serie_eventos_sp_sim_real_2010_2023.csv \
    --models tabpfn --horizon 6 --min-train-size 60 \
    --output-prefix results/revisao/tabpfn
```

`TabPFNForecaster` usa a mesma via recursiva do XGBoost e do CatBoost; o limitador de
taxa e a retentativa ficam em volta da chamada de rede, sem tocar no protocolo. Requer
`tabpfn_client` e o token em `TABPFN_TOKEN`. Custo: 103 ajustes e 618 predições na API.

`notebooks/conferencia_regeneracao.ipynb` roda isso no Colab e já aplica o teste pareado
em seguida. Duas cautelas que vêm da vez passada: montar o Drive, para que uma queda de
sessão não custe a hora de API outra vez, e rodar primeiro em modo de teste, que gasta
cinco janelas para provar que a chave funciona.

**Procedência do CSV versionado.** Ele saiu do `tabpfn_benchmark.ipynb`, que usa a porta
autossuficiente do protocolo, e não do `run_benchmark.py` acima, que usa o `skforecast` do
repositório. As duas vias foram feitas para coincidir, e a bancada do notebook se valida
reproduzindo `naive`, `snaive`, `snaive_drift` e `catboost` com diferença zero contra os
valores publicados. Ainda assim é uma via diferente da que gerou as demais linhas, e quem
rodar pelo `run_benchmark.py` deve comparar: se divergir, a divergência é resultado, não
erro de digitação.

**Cota da API.** O limite é diário, por conta, e contado em tokens, não em chamadas: 5
milhões por dia, com reset à meia-noite UTC. Uma rodada completa não cabe com folga, e a
de 23/09 precisou de dois dias de cota. O checkpoint é o que torna isso viável — sem ele,
esbarrar na cota no meio significa recomeçar do zero no dia seguinte.

## O que não entrou

`timesfm` e `tabpfn_ts` não foram rodados nessa sessão. O TimesFM já tem valor no artigo e
pede GPU. O TabPFN-TS é o enquadramento de série temporal em vez de tabular, e não é o que
o parecer pediu: o pedido era pelo competidor tabular do XGBoost e do CatBoost.
