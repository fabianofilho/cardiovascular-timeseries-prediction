# Manuscrito

O manuscrito vive em [`paper/manuscript.tex`](paper/manuscript.tex), compilado com:

```bash
cd paper && tectonic -X compile manuscript.tex --outdir ../build
```

Artefatos gerados a partir dele:

| Arquivo | O que é |
|---|---|
| `paper/manuscript.tex` | manuscrito, único arquivo editado à mão |
| `paper/tables/*.tex` | tabelas booktabs, geradas |
| `paper/figures/*.tex` | figuras em pgfplots, geradas, entram por `\input` |
| `paper/figures/*.png` | as mesmas figuras rasterizadas a 300 dpi, para conferir e para o Word |
| `paper/verified_numbers.json` | todo valor citado na prosa, para conferência |
| `paper/cover_letter.tex` | carta ao editor |
| `scripts/build_paper_assets.py` | gera tabelas, figuras e o JSON dos dados brutos |
| `scripts/build_docx.py` | gera o `.docx` de revisão para os coautores |

## Por que este arquivo não é mais o manuscrito

Até 17/08/2026 havia aqui uma cópia do manuscrito em Markdown, além da versão em LaTeX.
Duas cópias da mesma prosa divergiram em quatro dias, e a divergência não foi cosmética:
a versão em Markdown continuou afirmando que a política de exógena `lag12` tinha sido
"implemented and compared", quando ela nunca foi executada. A auditoria que pegou isso
corrigiu só o LaTeX, e a afirmação falsa seguiu viva aqui.

Manter dois manuscritos custa mais do que rende, e o custo aparece como afirmação errada
publicada, não como inconveniência. O LaTeX é o artefato de submissão, alimentado pelo
mesmo `verified_numbers.json` que gera as tabelas e as figuras; o `.docx` de revisão sai
dele pelo `build_docx.py`. Um só, com dois destinos.

O conteúdo antigo continua no histórico do git, no commit anterior a este.
