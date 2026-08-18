# Figuras superadas

Nada aqui entra no manuscrito. As figuras dele estão em `paper/figures/`, como código
pgfplots gerado por `scripts/build_paper_assets.py`, e os PNG de 300 dpi ao lado.

Estas ficam versionadas porque documentos históricos apontam para elas:
`docs/experiments/` e os task graphs registram rodadas que as usaram, e apagá-las
tornaria esse registro ilegível.

## Duas gerações, e um defeito conhecido

**Rodada de 2019-2023** (sem sufixo), de `images/legacy/generate_paper_figures.py`:
`fig1_time_series.png` a `fig7_seasonal_profile.png`.

Duas delas têm um defeito documentado: **`fig1_time_series.png` e
`fig7_seasonal_profile.png` contêm 24 meses sintetizados** para 2019 e 2020
(`generate_paper_figures.py:88-116`), rotulados como observação. Não reutilizar em lugar
nenhum. Foi esse achado que motivou a rodada seguinte.

**Rodada de 2010-2023** (sufixo `_2010_2023`), de `images/legacy/generate_figures_2010_2023.py`:
geradas só com dado observado, corrigindo o defeito acima. Foram as figuras do manuscrito
até 17/08/2026, quando a migração para pgfplots as substituiu por código.

## Os dois geradores, aposentados

`generate_paper_figures.py` e `generate_figures_2010_2023.py` saíram de `scripts/` e vivem
aqui, ao lado das figuras que produzem. Nenhum dos dois alimenta o manuscrito: quem faz isso
é `scripts/build_paper_assets.py`, que gera as figuras como código pgfplots.

Ficam no repositório em vez de serem apagados por um motivo específico: o registro do defeito
cita **`generate_paper_figures.py:88-116`** como a evidência dos meses sintetizados. Apagar o
arquivo tornaria essa alegação impossível de conferir, e uma alegação de defeito que ninguém
consegue verificar não vale nada.

Por isso a mudança neles foi mínima, uma linha só (`BASE`, para o caminho continuar
resolvendo do novo diretório): **a numeração de linha foi preservada de propósito**, para a
citação `88-116` continuar apontando para o mesmo bloco. É também por isso que o aviso de
depreciação está aqui, e não num cabeçalho dentro dos arquivos, que deslocaria as linhas.

Eles são a única coisa no repositório que ainda precisa de `matplotlib`. O pipeline atual usa
`pymupdf` e `tectonic` para rasterizar as figuras.
