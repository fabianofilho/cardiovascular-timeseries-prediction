# paper_v1b

Reescrita da v1 (IJF-D-26-01619, desk reject em 24/10/2026) com as regras de
`~/.claude/skills/paper/references/contribuicao-e-voz.md`. Mesmos dados e resultados da v1;
muda o texto, as legendas e o que fica no corpo. A v1 está intacta em `paper/` e na tag
`ijf-v1-2026-10-24`.

- `manuscript.tex`: texto principal (núcleo: comprimento do histórico importa mais que a
  escolha entre os três melhores modelos; o modelo de fundação só ganha com histórico curto).
- `supplement.tex`: ambiente de execução, variantes do boosting, Optuna, TabPFN, pandemia,
  temperatura com calendário. Entra no fim do PDF para as referências cruzadas.
- `build_tables.py`: pega o bloco numérico das tabelas geradas em `paper/tables` e troca só
  legenda e nota. Rodar de dentro de `paper_v1b/` sempre que `scripts/build_paper_assets.py`
  regenerar as tabelas.

Compilar da raiz do repo:

    (cd paper_v1b && python build_tables.py)
    tectonic -X compile paper_v1b/manuscript.tex --outdir build_v1b

Antes de mandar a coautores ou submeter: `python3 ~/.claude/skills/paper/scripts/marcas_ia.py
paper_v1b/manuscript.tex` sem falha, e o `[OPEN]` da introdução preenchido pelos autores.
