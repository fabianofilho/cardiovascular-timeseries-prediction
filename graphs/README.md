# graphs/

Página do grafo de execução do projeto, gerada pela fase RENDER da skill `graph-init`.

- **Arquivo:** `task-graph.html`
- **URL publicada:** https://claude.ai/code/artifact/31cb4718-af40-4ce5-b488-0aec3d4bc276
- **Fonte da verdade do grafo:** `task-graph.md`, na raiz do repositório
- **Fonte dos números:** os CSVs em `results/`, citados em cada figura da página

## Convenção

O nome do arquivo é estável. Toda rodada nova **reescreve `task-graph.html` e republica a mesma
URL**, em vez de criar arquivo novo. Quem recebeu o link antes precisa continuar caindo na versão
atual. A página diz, no rodapé, qual é a versão e o que mudou em relação à anterior.

Para republicar de outra sessão, passe a URL acima no parâmetro `url` ao publicar, senão uma URL
nova é criada e o link antigo congela numa versão velha.

## Versões

| versão | data | o que mudou |
|---|---|---|
| 1 | 01/08/2026 | primeira publicação. Rodada de incerteza do benchmark: IC por bootstrap de janelas, teste pareado, erro por horizonte, boosting na série reconstruída e SHAP dos lags |
