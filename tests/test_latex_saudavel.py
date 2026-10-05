"""Defeitos de LaTeX que nao quebram a compilacao e por isso passam despercebidos.

Todos os casos aqui ja aconteceram neste projeto, dois deles mais de uma vez. O padrao e
sempre o mesmo: o documento compila, o PDF sai, e o erro so aparece quando alguem le a
pagina. Um teste e o unico jeito de pegar isso antes da submissao.

1. `\\ref` escrito numa string nao-raw do Python: o `\\r` vira o byte 0x0D e o arquivo fica
   com "Table~<CR>ef{...}". Compila, e o PDF imprime "Table~ef{tab:...}".
2. `%` cru vindo de um format `:.1%`: em LaTeX o `%` comenta o resto da linha, e a nota da
   tabela perde a frase inteira dali em diante.
3. Comando de referencia apontando para um rotulo que nao existe em lugar nenhum.
4. Tabela ou figura gerada que nao e `\\input` em nenhum documento: o resultado existe no
   repositorio e nao chega ao leitor.
5. Simbolo que exige pacote entrando por regeneracao de figura, sem decisao de ninguem.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PAPER = RAIZ / "paper"
MANUSCRITO = PAPER / "manuscript.tex"

TEX = sorted(PAPER.rglob("*.tex"))
CONTROLE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
CR = chr(13)


def _texto(f: Path) -> str:
    return f.read_bytes().decode("utf-8", errors="replace")


@pytest.mark.parametrize("f", TEX, ids=lambda f: f.name)
def test_nenhum_caractere_de_controle_solto(f):
    """Assinatura do `\\ref` mutilado, e de qualquer outro escape que vazou.

    O PDF sai bonito com "Table~ef{tab:calibracao}" no meio do texto. Ja aconteceu duas
    vezes, uma em tab1_desempenho.tex e outra no manuscrito.
    """
    for i, linha in enumerate(_texto(f).split("\n"), 1):
        corpo = linha[:-1] if linha.endswith(CR) else linha
        assert not CONTROLE.search(corpo), f"{f.name}:{i} tem caractere de controle"
        assert CR not in corpo, (
            f"{f.name}:{i} tem carriage return no meio da linha. "
            "Provavelmente um '\\\\ref' escrito numa string nao-raw do Python.")


@pytest.mark.parametrize("f", TEX, ids=lambda f: f.name)
def test_nenhuma_porcentagem_crua(f):
    """`84.5%` comenta o resto da linha. Ja comeu uma frase inteira da Tabela 6."""
    for i, linha in enumerate(_texto(f).split("\n"), 1):
        for m in re.finditer(r"(?<!\\)%", linha):
            antes = linha[:m.start()]
            if antes.lstrip().startswith("%"):
                continue                      # comentario de linha, legitimo
            assert not antes.rstrip().endswith(tuple("0123456789")), (
                f"{f.name}:{i} tem '%' cru depois de digito: {linha.strip()[:70]!r}. "
                "Em LaTeX isso comenta o resto da linha; use '\\\\%'.")


def test_toda_referencia_aponta_para_um_rotulo_existente():
    """`Table ??` no PDF e um erro que so aparece lendo a pagina certa."""
    rotulos = set()
    for f in TEX:
        rotulos |= set(re.findall(r"\\label\{([^}]+)\}", _texto(f)))
    texto = _texto(MANUSCRITO)
    # so o que o manuscrito realmente inclui conta como disponivel
    for inc in re.findall(r"\\input\{([^}]+)\}", texto):
        alvo = PAPER / (inc + ".tex")
        if alvo.exists():
            rotulos |= set(re.findall(r"\\label\{([^}]+)\}", _texto(alvo)))
    usados = set(re.findall(r"\\(?:ref|autoref|cref)\{([^}]+)\}", texto))
    faltando = sorted(usados - rotulos)
    assert not faltando, f"referencias sem rotulo alcancavel: {faltando}"


def test_toda_tabela_gerada_entra_no_documento():
    """Resultado que nao e input nao chega ao leitor.

    Foi assim que a tabela do Optuna e a de temperatura calendarizada ficaram de fora: as
    duas eram geradas, tinham rotulo, e o estudo chegava ao leitor so em prosa.
    """
    incluidas = set(re.findall(r"\\input\{tables/([^}]+)\}", _texto(MANUSCRITO)))
    existentes = {f.stem for f in (PAPER / "tables").glob("*.tex")}
    fora = sorted(existentes - incluidas)
    assert not fora, (
        f"tabelas geradas e nunca incluidas no manuscrito: {fora}. "
        "Ou entram, ou saem do gerador; nao ficam no meio.")


# ---------------------------------------------------------------- simbolos e pacotes

# Um simbolo que exige pacote e o defeito perfeito deste projeto: nao quebra nada visivel
# enquanto o pacote estiver carregado, e quebra a compilacao inteira no dia em que alguem
# enxugar o preambulo. Foi o caso do \blacksquare, que entrou numa regeneracao de figuras
# sem que a troca tivesse sido decidida em lugar nenhum.
#
# A regra tem duas metades. A primeira cobra que todo simbolo de pacote usado tenha o
# pacote no preambulo. A segunda cobra que todo simbolo usado seja conhecido: um comando
# fora das duas listas reprova, e quem o introduziu tem de dizer de onde ele vem. Sem essa
# segunda metade a proxima troca de simbolo passaria igual a esta.

SIMBOLO_BASE = frozenset({
    # LaTeX base, sem pacote nenhum
    "bullet", "cdot", "cdots", "ldots", "dots", "times", "pm", "mp", "ast", "circ",
    "leq", "geq", "le", "ge", "neq", "ne", "approx", "sim", "equiv", "propto",
    "alpha", "beta", "gamma", "delta", "epsilon", "lambda", "mu", "sigma", "tau", "phi",
    "Delta", "Sigma", "Omega", "Gamma", "Lambda", "Phi",
    "frac", "sqrt", "sum", "prod", "int", "infty", "partial", "hat", "bar", "tilde",
    "left", "right", "quad", "qquad", "text", "mathrm", "mathbf", "mathit", "textcolor",
})

SIMBOLO_DE_PACOTE = {
    # simbolo -> pacote que o define
    "blacksquare": "amssymb",
    "square": "amssymb",
    "blacktriangle": "amssymb",
    "blacktriangledown": "amssymb",
    "blacklozenge": "amssymb",
    "lozenge": "amssymb",
    "checkmark": "amssymb",
    "vartriangle": "amssymb",
    "leqslant": "amssymb",
    "geqslant": "amssymb",
    "boldsymbol": "amsmath",
    "dfrac": "amsmath",
    "tfrac": "amsmath",
    "operatorname": "amsmath",
}

MODO_MATEMATICO = re.compile(r"\$([^$]*)\$")
COMANDO = re.compile(r"\\([a-zA-Z]+)")


def _sem_comentario(texto: str) -> str:
    """Tira comentario de linha, para nao contar simbolo citado em prosa de comentario."""
    linhas = []
    for linha in texto.split("\n"):
        m = re.search(r"(?<!\\)%", linha)
        linhas.append(linha[:m.start()] if m else linha)
    return "\n".join(linhas)


def _simbolos_usados() -> dict[str, set[str]]:
    """Comando -> nomes dos arquivos que o usam em modo matematico."""
    usados: dict[str, set[str]] = {}
    for f in TEX:
        for trecho in MODO_MATEMATICO.findall(_sem_comentario(_texto(f))):
            for cmd in COMANDO.findall(trecho):
                usados.setdefault(cmd, set()).add(f.name)
    return usados


def _pacotes_do_preambulo() -> set[str]:
    txt = _sem_comentario(_texto(PAPER / "preamble.tex"))
    return set(re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", txt))


def test_todo_simbolo_de_pacote_tem_o_pacote_carregado():
    """O \\blacksquare da fig4 veio assim: entrou na regeneracao e nao quebrou nada."""
    pacotes = _pacotes_do_preambulo()
    faltando = {}
    for cmd, arquivos in _simbolos_usados().items():
        pacote = SIMBOLO_DE_PACOTE.get(cmd)
        if pacote and pacote not in pacotes:
            faltando[cmd] = (pacote, sorted(arquivos))
    assert not faltando, (
        f"simbolos usados sem o pacote que os define: {faltando}. "
        "Ou o pacote entra no preamble.tex, ou o gerador volta ao simbolo de base.")


def test_nenhum_simbolo_desconhecido_em_modo_matematico():
    """Simbolo novo tem de ser declarado, em base ou em pacote, antes de entrar."""
    desconhecidos = {
        cmd: sorted(arquivos)
        for cmd, arquivos in _simbolos_usados().items()
        if cmd not in SIMBOLO_BASE and cmd not in SIMBOLO_DE_PACOTE
    }
    assert not desconhecidos, (
        f"simbolos nao catalogados: {desconhecidos}. "
        "Classifique cada um em SIMBOLO_BASE ou em SIMBOLO_DE_PACOTE, para que a proxima "
        "troca de simbolo numa regeneracao nao passe calada.")


# ------------------------------------------------------------------ paleta das figuras

def test_a_paleta_do_preambulo_e_a_do_gerador():
    """As figuras sao desenhadas duas vezes, e as duas tem de sair da mesma paleta.

    O PDF colore pelas definicoes do preamble.tex; o PNG que entra no .docx colore pelo
    dicionario COR do build_paper_assets.py, injetado no preambulo avulso da
    rasterizacao. Nada ligava os dois, e eles ficaram tres semanas divergentes nas cinco
    cores: o gerador migrou para Okabe-Ito e o preambulo continuou na paleta antiga.
    A mesma figura saia com cores diferentes conforme o formato, e a versao do PDF era
    a que confunde vermelho com verde sob deuteranopia.
    """
    gerador = (RAIZ / "scripts" / "build_paper_assets.py").read_text(encoding="utf-8")
    bloco = re.search(r"COR = \{(.*?)\}", gerador, re.S)
    assert bloco, "dicionario COR nao encontrado em build_paper_assets.py"
    do_gerador = dict(re.findall(r'"(\w+)":\s*"(\w+)"', bloco.group(1)))

    do_preambulo = dict(re.findall(r"\\definecolor\{c(\w+)\}\{HTML\}\{(\w+)\}",
                                   _texto(PAPER / "preamble.tex")))

    divergentes = {
        k: (do_preambulo.get(k), v) for k, v in do_gerador.items()
        if do_preambulo.get(k, "").upper() != v.upper()
    }
    assert not divergentes, (
        "paleta divergente entre preamble.tex e build_paper_assets.py "
        f"(preambulo, gerador): {divergentes}. A mesma figura sairia com cores "
        "diferentes no PDF e no .docx.")


# ---------------------------------------------------------------- figuras orfas

# Figuras que sao geradas de proposito sem entrar no manuscrito: elas respondem ao
# parecer na carta-resposta, nao no artigo. A lista existe para que "fora do manuscrito"
# seja uma decisao escrita e nao um esquecimento -- que foi o que aconteceu com as seis,
# geradas desde agosto, com rotulo, e nunca \input em lugar nenhum. O teste irmao cobria
# so tables/, entao nada reclamava.
#
# Tirar uma figura daqui e nao a incluir no manuscrito reprova. Incluir e nao tirar
# daqui tambem.
FIGURAS_FORA_DO_MANUSCRITO = {
    "revisao_fig_calibracao",
    "revisao_fig_enriched_janela",
    "revisao_fig_naive_estendida",
    "revisao_fig_optuna",
    "revisao_fig_temp_diffcal",
    "revisao_fig_variantes",
}


def test_toda_figura_gerada_ou_entra_no_documento_ou_esta_declarada():
    """Figura gerada que nao e input nao chega ao leitor, igual a tabela.

    A versao deste teste para tabelas ja existia e pegou duas. As figuras ficaram sem
    cobertura, e seis delas passaram um mes sendo geradas, rasterizadas e nunca lidas.
    """
    incluidas = set(re.findall(r"\\input\{figures/([^}]+)\}", _texto(MANUSCRITO)))
    existentes = {f.stem for f in (PAPER / "figures").glob("*.tex")}

    fora = sorted(existentes - incluidas - FIGURAS_FORA_DO_MANUSCRITO)
    assert not fora, (
        f"figuras geradas, nunca incluidas e nao declaradas: {fora}. "
        "Ou entram no manuscrito, ou entram em FIGURAS_FORA_DO_MANUSCRITO com o motivo.")

    declaradas_mas_incluidas = sorted(FIGURAS_FORA_DO_MANUSCRITO & incluidas)
    assert not declaradas_mas_incluidas, (
        f"declaradas como fora do manuscrito e mesmo assim incluidas: "
        f"{declaradas_mas_incluidas}. Tire-as da lista.")

    declaradas_inexistentes = sorted(FIGURAS_FORA_DO_MANUSCRITO - existentes)
    assert not declaradas_inexistentes, (
        f"declaradas na lista e nao geradas por ninguem: {declaradas_inexistentes}.")
