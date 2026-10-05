#!/usr/bin/env python3
"""Gera o notebook do Colab que confere a regeneracao do XGBoost de ponta a ponta.

Por que um notebook e nao um script: as quatro conferencias precisam do ambiente do
lock, e a maquina de quem confere nao e a maquina que rodou. O Colab da um ambiente
limpo a cada execucao, que e a condicao que faltava da ultima vez -- a divergencia do
SARIMA que passou meses atribuida a otimo local do otimizador era versao de biblioteca.

Por que gerar em vez de escrever a mao: os numeros que o notebook compara vem das
fontes do repositorio, nao de digitacao. O gerador injeta os caminhos e as referencias;
o notebook le tudo do clone que ele mesmo faz.

O que o notebook confere:
    1. simbolos de LaTeX que a regeneracao de figuras trocou, e o pacote que eles exigem
    2. de que par sao as colunas de IC e DM da tabela de variantes
    3. se a rodada de calibracao versionada reproduz sob semente fixa
    4. o teste pareado do TabPFN, que exige a previsao janela a janela
    5. a conferencia geral: testes, assets regerados e diff do paper

Saida:
    notebooks/conferencia_regeneracao.ipynb

Uso:
    python scripts/build_conferencia_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nbtools import caderno, code, md, valida  # noqa: E402

SAIDA = ROOT / "notebooks" / "conferencia_regeneracao.ipynb"

# O notebook versionado e o do repositorio; a copia no Drive e so o que se abre no
# Colab. Ela e reescrita a cada geracao para as duas nao divergirem sem ninguem ver,
# que e como uma analise acaba rodando de uma versao que ninguem consegue achar depois.
DRIVE = Path(
    r"G:\.shortcut-targets-by-id\1zRZlDXqjBRVUlO0Zys2L0W69eJvgMI58\Labs"
    r"\Cardiovascular Time Series\IJF_Series_Temporais_CV\02_experimentos\notebooks"
)

# A regeneracao a conferir, e a arvore anterior a branch, que serve de "antes" para o
# inventario de simbolos.
REF_DEPOIS = "e829dc2"
REF_ANTES_BRANCH = "627a924"
REF_ANTES_REGEN = "792fe6c"

SERIE_CSV = "results/series/serie_eventos_sp_sim_real_2010_2023.csv"


# --------------------------------------------------------------------- celulas

CEL_SETUP = '''
# Clona o repositorio e entra no commit que se quer conferir.
#
# O repositorio e publico, entao o clone nao pede credencial nenhuma. Token so faz
# falta para a ultima celula, que envia o resultado, e ela e opcional. Se um segredo
# GITHUB_TOKEN existir nos Secrets do Colab (icone de chave na barra lateral), ele e
# usado; senao o clone segue anonimo. Token nunca vai colado numa celula: o notebook
# fica salvo com o que estiver escrito nele.
import os, subprocess, sys, json, re, shutil
from pathlib import Path

# O fork, e nao o repositorio de origem: o que a conferencia roda -- os testes de
# simbolo, o forecaster do TabPFN, o checkpoint por janela -- ainda nao esta no main.
# A arvore conferida continua sendo a da regeneracao, que e ancestral desta branch.
REPO = "isasaade-23/cardiovascular-timeseries-prediction"  #@param {type:"string"}
REF = "isasaade/conferencia-regeneracao"  #@param {type:"string"}
DESTINO = "/content/repo"

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    try:
        from google.colab import userdata
        TOKEN = userdata.get("GITHUB_TOKEN")
    except Exception:
        TOKEN = None

# Sai do destino ANTES de apaga-lo. Rodar esta celula duas vezes na mesma sessao
# deixava o processo dentro de um diretorio recem-apagado, e o git falhava com 128
# sem dizer por que: ele nao consegue nem ler o proprio diretorio de trabalho.
os.chdir("/content")
if Path(DESTINO).exists():
    shutil.rmtree(DESTINO)

if TOKEN:
    url = f"https://x-access-token:{TOKEN.strip()}@github.com/{REPO}.git"
    print("clone autenticado (o envio da ultima celula fica disponivel)")
else:
    url = f"https://github.com/{REPO}.git"
    print("clone anonimo: da para conferir tudo, menos enviar na ultima celula")

# Sem --quiet, e com a saida de erro na tela: um clone que falha calado custa mais
# tempo do que o ruido de um clone que fala.
for cmd in (["git", "clone", url, DESTINO],
            ["git", "-C", DESTINO, "checkout", REF]):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        visivel = " ".join(cmd).replace(TOKEN.strip(), "***") if TOKEN else " ".join(cmd)
        raise SystemExit(f"falhou: {visivel}\\n{r.stderr.strip()}")
os.chdir(DESTINO)
sys.path.insert(0, str(Path(DESTINO) / "src"))
os.environ["PYTHONPATH"] = str(Path(DESTINO) / "src")

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
print(f"repositorio em {DESTINO}, HEAD {sha}")

# VEREDITOS acumula o resultado de cada secao. A ultima celula escreve o relatorio a
# partir daqui, para que o texto nao possa discordar do que as celulas mediram.
VEREDITOS = {}
'''

CEL_AMBIENTE = '''
# Instala o ambiente fixado. Demora, e e o ponto do exercicio: e o statsmodels 0.15.0
# que faz o SARIMA reproduzir; com 0.14.6 ele diverge em duas das 103 janelas.
#
# O `tail` de antes escondia justamente o que importa. O pip resolve conflito
# desistindo em silencio de rebaixar o que o Colab ja traz instalado, e o resultado e
# um ambiente que parece o do lock e nao e. A saida inteira vai para arquivo, e as
# linhas de conflito sobem para a tela.
!pip install -r requirements-lock.txt > /content/pip.log 2>&1
conflitos = [l for l in Path("/content/pip.log").read_text().splitlines()
             if any(p in l.lower() for p in
                    ("error", "conflict", "incompatible", "cannot install"))]
if conflitos:
    print(f"  {len(conflitos)} linha(s) de conflito na instalacao:")
    for l in conflitos[:15]:
        print("   ", l[:140])
else:
    print("  instalacao sem conflito declarado")
'''

CEL_VERSOES = '''
# Confere o que ficou instalado contra o que o lock pede. Divergencia aqui explica
# divergencia numerica mais adiante, e e a primeira coisa a olhar quando um numero nao
# bate -- foi o que aconteceu com o XGBoost por sete versoes seguidas.
import importlib.metadata as md_

lock = {}
for linha in Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines():
    linha = linha.split("#")[0].strip()
    if "==" in linha:
        nome, versao = linha.split("==", 1)
        lock[nome.strip().lower()] = versao.strip()

IMPORTA = ["statsmodels", "xgboost", "catboost", "prophet", "numpy", "pandas",
           "scikit-learn", "scipy", "skforecast"]
divergentes = {}
print(f"  {'pacote':<16}{'instalado':>12}{'no lock':>12}")
for nome in IMPORTA:
    try:
        posta = md_.version(nome)
    except md_.PackageNotFoundError:
        posta = "ausente"
    quer = lock.get(nome.lower(), "-")
    marca = "" if posta == quer else "   <-"
    if posta != quer:
        divergentes[nome] = (posta, quer)
    print(f"  {nome:<16}{posta:>12}{quer:>12}{marca}")

VEREDITOS["ambiente"] = {
    "divergentes": divergentes,
    "conclusao": (
        "ambiente igual ao lock: divergencia numerica daqui para frente e do codigo, "
        "nao da bancada." if not divergentes else
        f"{len(divergentes)} pacote(s) fora do lock: {sorted(divergentes)}. Enquanto "
        "isso valer, nenhuma divergencia numerica pode ser atribuida ao codigo, "
        "porque a bancada tambem mudou."),
}
print("\\n  " + VEREDITOS["ambiente"]["conclusao"])

if divergentes:
    print("\\n  " + "=" * 66)
    print("  O Colab nao consegue rebaixar numpy, pandas e afins com a sessao ja")
    print("  em pe: os modulos estao carregados. Reinicie o ambiente de execucao")
    print("  (Ambiente de execucao > Reiniciar sessao) e rode de novo a partir da")
    print("  primeira celula. A instalacao ja esta no disco e passa rapido.")
    print("  " + "=" * 66)
'''

CEL_SIMBOLOS = '''
# SECAO 1 -- simbolos de LaTeX
#
# A pergunta e se a regeneracao das figuras trocou simbolos alem do \\blacksquare que
# ja foi corrigido. O metodo e inventario, nao leitura: extrai todo comando em modo
# matematico de cada .tex, nas duas arvores, e compara.
MATH = re.compile(r"\\$([^$]*)\\$")
CMD = re.compile(r"\\\\([a-zA-Z]+)")

def sem_comentario(texto):
    saida = []
    for linha in texto.split("\\n"):
        m = re.search(r"(?<!\\\\)%", linha)
        saida.append(linha[:m.start()] if m else linha)
    return "\\n".join(saida)

def inventario(ref):
    """Comando em modo matematico -> arquivos que o usam, na arvore de `ref`."""
    listagem = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref],
                              capture_output=True, text=True, check=True).stdout
    achados = {}
    for caminho in listagem.splitlines():
        if not (caminho.startswith("paper/") and caminho.endswith(".tex")):
            continue
        texto = subprocess.run(["git", "show", f"{ref}:{caminho}"],
                               capture_output=True, text=True, check=True).stdout
        for trecho in MATH.findall(sem_comentario(texto)):
            for cmd in CMD.findall(trecho):
                achados.setdefault(cmd, set()).add(caminho.split("/")[-1])
    return achados

ANTES, DEPOIS = "627a924", "e829dc2"
antes, depois = inventario(ANTES), inventario(DEPOIS)

print(f"  arvore anterior a branch ({ANTES}):")
for cmd in sorted(antes):
    print(f"    \\\\{cmd:<14} {sorted(antes[cmd])}")
print(f"\\n  arvore atual ({DEPOIS}):")
for cmd in sorted(depois):
    print(f"    \\\\{cmd:<14} {sorted(depois[cmd])}")

entraram = sorted(set(depois) - set(antes))
sairam = sorted(set(antes) - set(depois))
print(f"\\n  entraram com a regeneracao: {entraram or 'nenhum'}")
print(f"  sairam: {sairam or 'nenhum'}")
'''

CEL_SIMBOLOS_PACOTE = '''
# Cada simbolo que entrou exige pacote? E o preambulo carrega esse pacote?
DE_PACOTE = {"blacksquare": "amssymb", "square": "amssymb", "checkmark": "amssymb",
             "blacktriangle": "amssymb", "lozenge": "amssymb", "leqslant": "amssymb",
             "geqslant": "amssymb", "boldsymbol": "amsmath", "dfrac": "amsmath"}

preambulo = Path("paper/preamble.tex").read_text(encoding="utf-8")
pacotes = set(re.findall(r"\\\\usepackage(?:\\[[^\\]]*\\])?\\{([^}]+)\\}",
                         sem_comentario(preambulo)))

exigem, faltando = {}, {}
for cmd, arquivos in depois.items():
    pacote = DE_PACOTE.get(cmd)
    if not pacote:
        continue
    exigem[cmd] = (pacote, sorted(arquivos))
    if pacote not in pacotes:
        faltando[cmd] = pacote

print("  simbolos que exigem pacote, na arvore atual:")
for cmd, (pacote, arquivos) in sorted(exigem.items()):
    print(f"    \\\\{cmd:<14} {pacote:<10} {arquivos}")
print(f"\\n  pacotes no preambulo: {sorted(pacotes)}")
print(f"  sem o pacote correspondente: {faltando or 'nenhum'}")

arquivos_dependentes = sorted({a for _, arqs in exigem.values() for a in arqs})
VEREDITOS["simbolos"] = {
    "entraram_com_a_regeneracao": entraram,
    "exigem_pacote": {k: v[0] for k, v in exigem.items()},
    "arquivos_dependentes": arquivos_dependentes,
    "sem_pacote": faltando,
    "conclusao": (
        f"{len(arquivos_dependentes)} arquivo(s) dependem de pacote de simbolo "
        f"({', '.join(arquivos_dependentes)}); "
        + ("todos com o pacote carregado" if not faltando
           else f"SEM o pacote: {faltando}")),
}
print("\\n  " + VEREDITOS["simbolos"]["conclusao"])
'''

CEL_SIMBOLOS_TESTE = '''
# O inventario acima e uma foto. O teste e o que impede a proxima regeneracao de trocar
# um simbolo sem ninguem ver: ele reprova simbolo de pacote sem pacote, e reprova
# tambem simbolo nao catalogado, que e como um simbolo novo entra sem decisao.
!python -m pytest tests/test_latex_saudavel.py -q 2>&1 | tail -5
'''

CEL_TAB7 = '''
# SECAO 2 -- de que par sao as colunas de IC e DM da tabela de variantes
#
# Na regeneracao, os pontos do XGBoost na tabela mudaram e as colunas de IC e DM
# ficaram identicas. Ou a tabela ficou com numero velho, ou essas colunas nunca foram
# do XGBoost. A resposta esta no gerador, nao na memoria de ninguem.
gerador = Path("scripts/build_paper_assets.py").read_text(encoding="utf-8")
bloco = gerador[gerador.index("Tabela 7"):]
bloco = bloco[:bloco.index("escreve_tabela")]
linhas_fonte = [l for l in bloco.split("\\n")
                if "ic_low" in l or "dm_significativos" in l or "vj[" in l]
print("  o que o gerador poe nessas duas colunas:")
for l in linhas_fonte:
    print("   ", l.strip())

vj = json.loads(Path("results/revisao/variants_vs_snaive.json").read_text(encoding="utf-8"))
tex = Path("paper/tables/tab7_variantes.tex").read_text(encoding="utf-8")

print("\\n  IC e DM de cada modelo, medidos, para a variante base:")
for m in ("catboost_base", "xgboost_base"):
    d = vj["modelos"][m]
    print(f"    {m:<16} [{d['ic_low']:+.2f}, {d['ic_high']:+.2f}]  "
          f"{d['dm_significativos']}/6")

na_tabela = re.search(r"Lags only \\(as reported\\).*", tex).group(0)
print(f"\\n  a linha na tabela: {na_tabela.strip()}")

do_catboost = f"[{vj['modelos']['catboost_base']['ic_low']:+.2f}, "\\
              f"{vj['modelos']['catboost_base']['ic_high']:+.2f}]"
casa_catboost = do_catboost in na_tabela
VEREDITOS["tab7"] = {
    "coluna_e_do_par": "catboost vs snaive" if casa_catboost else "indeterminado",
    "ic_catboost_base": do_catboost,
    "confere": bool(casa_catboost),
    "conclusao": (
        "as colunas de IC e DM sao do par CatBoost contra naive sazonal, como o "
        "cabecalho e a legenda ja dizem. Nao envolvem o XGBoost, entao regenerar o "
        "XGBoost nao devia move-las: a tabela esta certa."
        if casa_catboost else
        "o IC da tabela nao casa com o do CatBoost medido. Regerar os assets e "
        "comparar de novo antes de qualquer conclusao."),
}
print("\\n  " + VEREDITOS["tab7"]["conclusao"])
'''

CEL_CALIBRACAO = '''
# SECAO 3 -- qual rodada de calibracao e canonica
#
# O script ja semeia. Se a rodada versionada reproduzir sob a semente, ela e a
# canonica e o item fecha. Se nao reproduzir, a semeada passa a ser, e a figura e a
# tabela que dependem dela tem de ser regeradas da mesma rodada.
import pandas as pd

VERS_CSV = Path("results/calibracao_2010_2023_predictions.csv")
VERS_JSON = Path("results/calibracao_2010_2023_metrics.json")
guardado_csv = pd.read_csv(VERS_CSV)
guardado_json = json.loads(VERS_JSON.read_text(encoding="utf-8"))

!python scripts/run_calibracao.py 2>&1 | tail -12

novo_csv = pd.read_csv(VERS_CSV)
novo_json = json.loads(VERS_JSON.read_text(encoding="utf-8"))
'''

CEL_CALIBRACAO_DIFF = '''
# Comparacao numerica, coluna a coluna. Hash de arquivo nao serve: um numero que muda
# na quinta casa e uma coisa, uma coluna inteira deslocada e outra, e as duas dao hash
# diferente.
assert list(guardado_csv.columns) == list(novo_csv.columns), "colunas mudaram"
assert len(guardado_csv) == len(novo_csv), "numero de linhas mudou"

maximos = {}
for col in guardado_csv.columns:
    if guardado_csv[col].dtype.kind not in "fi":
        iguais = bool((guardado_csv[col] == novo_csv[col]).all())
        maximos[col] = 0.0 if iguais else float("nan")
        continue
    maximos[col] = float((guardado_csv[col] - novo_csv[col]).abs().max())

print(f"  {'coluna':<14}{'maior diferenca absoluta':>26}")
for col, v in maximos.items():
    print(f"  {col:<14}{v:>26.6f}")

# A distincao que decide o item: previsao pontual e uma coisa, banda e outra. As duas
# movendo juntas e sinal de bancada diferente. So a banda movendo, com o ponto parado
# na ultima casa, e sinal de amostragem nao semeada -- e ai o problema esta no codigo,
# nao na maquina.
PONTUAIS = [c for c in ("y_pred", "y_true") if c in maximos]
BANDAS = [c for c in ("lo", "hi", "largura", "is", "dentro") if c in maximos]

moveu_ponto = any(maximos[c] != 0.0 for c in PONTUAIS)
moveu_banda = any(maximos[c] != 0.0 for c in BANDAS)
reproduz = not moveu_ponto and not moveu_banda
ambiente_sujo = bool(VEREDITOS.get("ambiente", {}).get("divergentes"))

if ambiente_sujo:
    conclusao = (
        "indeterminado. A rodada nao reproduziu, mas o ambiente nao e o do lock "
        f"({sorted(VEREDITOS['ambiente']['divergentes'])}), entao a divergencia pode "
        "ser da bancada e nao da semente. Reinicie a sessao para o lock valer e rode "
        "esta secao de novo: este item so fecha com os dois lados iguais.")
elif reproduz:
    conclusao = ("a rodada versionada reproduz sob a semente fixa, no ambiente do "
                 "lock: ela e a canonica, e o que faltava era declarar isso.")
elif moveu_banda and not moveu_ponto:
    conclusao = (
        "a previsao pontual reproduz exata e so a banda se move, no ambiente do lock. "
        "A semente de run_calibracao.py nao alcanca a amostragem que produz o "
        "intervalo, entao 'rodada canonica' nao se resolve escolhendo uma das "
        "rodadas: ou a amostragem passa a ser semeada de verdade, ou o texto para de "
        "dizer que ela e semeada e a rodada versionada e declarada a de referencia.")
else:
    conclusao = (
        "a previsao pontual se move, no ambiente do lock. Isso e mais grave que "
        "calibracao: o ponto e o que a Tabela 1 reporta, e ele nao devia depender de "
        "rodada. Antes de escolher rodada canonica, achar o que move o ponto.")

VEREDITOS["calibracao"] = {
    "reproduz_byte_a_byte": bool(reproduz),
    "moveu_previsao_pontual": bool(moveu_ponto),
    "moveu_banda": bool(moveu_banda),
    "ambiente_do_lock": not ambiente_sujo,
    "maior_diferenca_por_coluna": maximos,
    "picp_guardado": {m: guardado_json.get(m, {}).get("picp")
                      for m in ("sarima", "prophet") if m in guardado_json},
    "picp_novo": {m: novo_json.get(m, {}).get("picp")
                  for m in ("sarima", "prophet") if m in novo_json},
    "conclusao": conclusao,
}
print("\\n  " + conclusao)
'''

CEL_CALIBRACAO_SEMENTE = '''
# A semente de run_calibracao.py alcanca a amostragem do Prophet, ou nao?
#
# A pergunta e respondivel em dois ajustes, e nao depende de versao de biblioteca nem
# de arqueologia de changelog: semeia, ajusta, semeia igual, ajusta de novo, compara.
# Se o ponto sai identico e a banda nao, a semente nao chega onde produz o intervalo.
# O manuscrito ja afirma isso em prosa, na nota da Tabela 6; aqui vira medicao.
import numpy as np
sys.path.insert(0, "scripts")
from run_calibracao import prophet_intervalo, SEED
from cv_timeseries.data import load_and_aggregate_series

serie = load_and_aggregate_series(
    "SERIE_CSV_PLACEHOLDER", "date", "value", "MS")
treino = serie.iloc[:60]

saidas = []
for _ in range(2):
    np.random.seed(SEED)
    saidas.append(prophet_intervalo(treino, 6))

d_ponto = float(np.abs(saidas[0][0] - saidas[1][0]).max())
d_lo = float(np.abs(saidas[0][1] - saidas[1][1]).max())
d_hi = float(np.abs(saidas[0][2] - saidas[1][2]).max())
print(f"  dois ajustes da MESMA janela, com a mesma semente antes de cada um:")
print(f"    maior diferenca no ponto : {d_ponto:.6f}")
print(f"    maior diferenca no limite inferior: {d_lo:.6f}")
print(f"    maior diferenca no limite superior: {d_hi:.6f}")

alcanca = d_lo == 0.0 and d_hi == 0.0
VEREDITOS["semente_do_prophet"] = {
    "diferenca_ponto": d_ponto, "diferenca_lo": d_lo, "diferenca_hi": d_hi,
    "semente_alcanca_a_banda": bool(alcanca),
    "conclusao": (
        "a semente alcanca a banda: duas rodadas seguidas dao o mesmo intervalo, e "
        "entao divergencia entre rodadas e de ambiente, nao de amostragem."
        if alcanca else
        "a semente NAO alcanca a banda: dois ajustes da mesma janela, com a mesma "
        "semente, dao intervalos diferentes. `np.random.seed` nao controla o gerador "
        "que produz o intervalo nesta versao do Prophet. Enquanto isso valer, nenhuma "
        "rodada de calibracao e reproduzivel, e escolher uma como canonica so fixa um "
        "numero sem torna-lo verificavel."),
}
print("\\n  " + VEREDITOS["semente_do_prophet"]["conclusao"])
'''

CEL_TABPFN_MODO = '''
# SECAO 4 -- TabPFN por janela
#
# O que falta nao e o numero agregado, que ja existe: e a previsao janela a janela, sem
# a qual o criterio pre-declarado (intervalo de bootstrap pareado excluindo zero E
# Diebold-Mariano p<0,05 em ao menos 3 de 6 horizontes) nao pode ser calculado.
#
# A rodada sai pelo mesmo caminho que gerou todas as outras linhas do artigo:
# run_benchmark.py com o forecaster que usa a via recursiva dos boosters. Rodar por
# fora e trazer so o agregado foi o que deixou este item em aberto.
#
# Custo da rodada completa: 103 ajustes e 618 predicoes na API, pouco mais de uma hora.

MODO = "nao rodar"  #@param ["nao rodar", "teste (5 janelas)", "completo (103 janelas)", "usar CSV ja pronto"]

ALVO = Path("results/revisao/tabpfn_predictions.csv")

if MODO == "usar CSV ja pronto":
    from google.colab import files
    print("Envie o CSV com colunas model,window,horizon,date,y_true,y_pred")
    enviados = files.upload()
    nome = list(enviados)[0]
    ALVO.parent.mkdir(parents=True, exist_ok=True)
    Path(nome).replace(ALVO)
    print(f"  {ALVO} recebido")
elif MODO != "nao rodar":
    !pip -q install tabpfn-client
    # A chave vem dos Secrets do Colab, do segredo PRIOR_LABS_TOKEN. Nao ha prompt de
    # digitacao aqui de proposito: chave digitada numa celula fica no notebook salvo.
    # A biblioteca le a variavel TABPFN_TOKEN, entao o segredo alimenta a variavel.
    # Variantes numeradas entram na busca porque a cota e diaria e por conta: quando
    # ela esgota no meio de uma rodada, a saida e uma chave de outra conta, e o
    # segredo novo acaba nomeado PRIOR_LABS_TOKEN2, 3, e assim por diante. A mais
    # recente primeiro. Os Secrets vem antes de os.environ de proposito: na ordem
    # inversa, a chave da execucao anterior sobrevive a troca do segredo.
    from google.colab import userdata
    token, de_onde = None, None
    for nome_segredo in [f"PRIOR_LABS_TOKEN{n}" for n in ("5", "4", "3", "2", "")]:
        try:
            token = userdata.get(nome_segredo)
        except Exception:
            token = None
        if token:
            de_onde = nome_segredo
            break
    if not token:
        raise SystemExit(
            "Nenhum segredo PRIOR_LABS_TOKEN com valor e com acesso para este "
            "notebook. No icone de chave da barra lateral, confira o valor e ligue "
            "o acesso.")
    os.environ["TABPFN_TOKEN"] = token.strip()
    # Os ultimos caracteres, para conferir de olho QUAL chave esta em uso.
    print(f"  chave de {de_onde}, terminando em "
          f"...{os.environ['TABPFN_TOKEN'][-6:]}")

    # Checkpoint no Drive, e nao no disco da sessao. Uma rodada de mais de uma hora
    # que cai no minuto 50 sem isto custa a hora inteira, e ja custou uma vez: foi
    # assim que a unica medicao por janela que existia se perdeu. Retomar e so rodar
    # esta celula de novo; as janelas ja feitas nao voltam para a API.
    # force_remount resolve o "mount failed" mais comum, que e uma montagem anterior
    # meio morta na mesma sessao. Sem Drive a rodada ainda e retomavel dentro da
    # sessao; o que se perde e a sobrevivencia a uma queda dela.
    if not os.path.isdir("/content/drive/MyDrive"):
        for forcar in (False, True):
            try:
                from google.colab import drive
                drive.mount("/content/drive", force_remount=forcar)
                break
            except Exception as e:
                print(f"  Drive nao montou (force_remount={forcar}): {e}")
    if os.path.isdir("/content/drive/MyDrive"):
        CHECKPOINT = "/content/drive/MyDrive/tabpfn_cv/checkpoint.csv"
    else:
        CHECKPOINT = "/content/tabpfn_checkpoint.csv"
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    print(f"  checkpoint em {CHECKPOINT}")

    # No modo de teste o horizonte e o minimo de treino ficam iguais; o que muda e o
    # tamanho da serie, cortada para dar poucas janelas. Um sMAPE de 5 janelas NAO se
    # compara com o das 103, e por isso ele nao entra em lugar nenhum: serve para
    # provar que a chave e a via funcionam antes de gastar a hora.
    entrada = "SERIE_CSV_PLACEHOLDER"
    if MODO.startswith("teste"):
        import pandas as pd
        d = pd.read_csv(entrada).head(71)
        entrada = "/content/serie_teste.csv"
        d.to_csv(entrada, index=False)
        print(f"  modo de teste: {len(d)} meses, {len(d) - 60 - 6 + 1} janelas")

    # No modo de teste o checkpoint nao entra: ele guarda janelas da rodada completa,
    # e misturar as duas daria um resultado que nao e de nenhuma das duas.
    ck = "" if MODO.startswith("teste") else CHECKPOINT
    # Sem `tail`: numa rodada de mais de uma hora, filtrar a saida significa uma hora
    # de tela vazia, em que travado e rodando sao a mesma coisa. O script imprime uma
    # linha por janela com o tempo restante estimado.
    !python scripts/run_benchmark.py --input-csv "$entrada" --models tabpfn --horizon 6 --min-train-size 60 --output-prefix results/revisao/tabpfn --checkpoint-csv "$ck"
else:
    print("Nada a rodar. Para produzir o CSV, troque MODO acima.")
    print("Sem ele a secao seguinte pula o TabPFN e o item continua em aberto.")

print(f"\\n  CSV presente: {ALVO.exists()}")
'''

CEL_TABPFN_TESTE = '''
# O teste pareado. Roda para todo mundo de uma vez, com as MESMAS janelas reamostradas
# e a mesma semente: e o que torna os intervalos comparaveis entre si.
!python scripts/analisa_variantes.py 2>&1 | tail -30
'''

CEL_TABPFN_VEREDITO = '''
# Veredito do TabPFN contra as duas referencias ingenuas que importam. O criterio do
# artigo compara com o naive sazonal; a afirmacao que ficou em aberto era mais forte,
# de bater as referencias ingenuas, entao o naive sazonal com drift entra tambem.
import numpy as np
sys.path.insert(0, "scripts")
from analisa_variantes import dm_test, smape_vec, matriz, B, SEED

if not ALVO.exists():
    VEREDITOS["tabpfn"] = {
        "conclusao": "sem CSV por janela, o criterio pre-declarado continua nao "
                     "calculavel e a afirmacao segue sem lastro. Item em aberto."}
    print("  " + VEREDITOS["tabpfn"]["conclusao"])
else:
    import pandas as pd
    todos = pd.concat([
        pd.read_csv("results/revisao/variants_predictions.csv"),
        pd.read_csv("results/benchmark_baselines_2010_2023_predictions.csv"),
        pd.read_csv("results/benchmark_sim_real_sp_2010_2023_predictions.csv"),
        pd.read_csv(ALVO)], ignore_index=True)

    sm, ae = {}, {}
    for m in ("tabpfn", "snaive", "snaive_drift", "naive", "catboost_direct"):
        try:
            yt, yp = matriz(todos, m)
        except Exception as e:
            print(f"  {m}: {e}")
            continue
        sm[m], ae[m] = smape_vec(yt, yp), np.abs(yp - yt)

    rng = np.random.default_rng(SEED)
    nw = sm["snaive"].shape[0]
    idx = rng.integers(0, nw, size=(B, nw))

    resultado = {}
    for ref in ("snaive", "snaive_drift", "naive"):
        if "tabpfn" not in sm or ref not in sm:
            continue
        dif = sm["tabpfn"][idx].mean(axis=(1, 2)) - sm[ref][idx].mean(axis=(1, 2))
        lo, hi = np.percentile(dif, [2.5, 97.5])
        sig = 0
        for h in range(6):
            _, p = dm_test(ae["tabpfn"][:, h] - ae[ref][:, h], h + 1)
            if p == p and p < 0.05:
                sig += 1
        passa = bool(hi < 0 and sig >= 3)
        resultado[ref] = {"delta_pp": float(sm["tabpfn"].mean() - sm[ref].mean()),
                          "ic": [float(lo), float(hi)], "dm": f"{sig}/6",
                          "atende_criterio": passa}
        print(f"  tabpfn vs {ref:<14} delta {resultado[ref]['delta_pp']:+.4f} pp   "
              f"IC [{lo:+.3f}, {hi:+.3f}]   DM {sig}/6   "
              f"{'ATENDE' if passa else 'nao atende'}")

    atende = [r for r, d in resultado.items() if d["atende_criterio"]]
    VEREDITOS["tabpfn"] = {
        "smape": float(sm["tabpfn"].mean()),
        "por_referencia": resultado,
        "conclusao": (
            f"o TabPFN atende o criterio pre-declarado contra {', '.join(atende)}."
            if atende else
            "o TabPFN nao atende o criterio pre-declarado contra nenhuma das "
            "referencias ingenuas. O que se pode afirmar e que ele e o melhor modelo "
            "tabular testado, nao que ele bate uma regra sem modelo."),
    }
    print("\\n  " + VEREDITOS["tabpfn"]["conclusao"])
'''

CEL_CONFERENCIA = '''
# SECAO 5 -- conferencia geral
#
# Testes primeiro. Se algum falhar aqui e nao falhar por causa de alguma secao acima,
# e achado: o main estaria quebrado desde antes.
!python -m pytest -q 2>&1 | tail -15
'''

CEL_CONFERENCIA_ASSETS = '''
# Regera os assets e olha o diff. Tabela ou figura que mude sem que uma das secoes
# acima explique a mudanca e achado, nao ruido. O inverso tambem vale: nenhuma
# mudanca depois de uma regeneracao aplicada quer dizer que ela nao chegou ao paper.
!python scripts/build_paper_assets.py 2>&1 | tail -8
print("\\n  diff em paper/ depois de regerar:")
!git diff --stat -- paper/ | tail -20

diff = subprocess.run(["git", "diff", "--name-only", "--", "paper/"],
                      capture_output=True, text=True).stdout.split()

# A secao 3 reescreve os CSVs da calibracao, e a tabela 6 e os numeros verificados
# saem deles. Esse diff e consequencia dela, nao achado novo, e dizer o contrario
# transformaria um efeito colateral conhecido em alarme.
DA_CALIBRACAO = {"paper/tables/tab6_calibracao.tex", "paper/verified_numbers.json"}
esperado = sorted(set(diff) & DA_CALIBRACAO)
inesperado = sorted(set(diff) - DA_CALIBRACAO)

VEREDITOS["assets"] = {
    "arquivos_com_diff": diff,
    "explicados_pela_calibracao": esperado,
    "sem_explicacao": inesperado,
    "conclusao": (
        "regerar os assets nao move nada no paper: o que esta versionado e o que a "
        "fonte produz." if not diff else
        (f"regerar move {len(diff)} arquivo(s). {esperado} sai da rodada de "
         "calibracao desta sessao, que e efeito da secao 3 e nao achado."
         + (f" Sem explicacao: {inesperado}, e cada um precisa de uma antes de ser "
            "commitado." if inesperado else
            " Nada mais se move, entao a regeneracao do XGBoost esta inteira no que "
            "ja foi commitado."))),
}
print("\\n  " + VEREDITOS["assets"]["conclusao"])
'''

CEL_CONFERENCIA_ABERTOS = '''
# O que o manuscrito ainda declara em aberto. Sao marcas deliberadas: valem como lista
# de pendencias, e nao como defeito.
texto = Path("paper/manuscript.tex").read_text(encoding="utf-8")
abertos = re.findall(r"\\\\(?:missing|aberto)\\{([^}]{0,120})", texto)
print(f"  {len(abertos)} marca(s) em aberto no manuscrito:")
for a in abertos:
    print("   -", " ".join(a.split())[:100])
VEREDITOS["abertos_no_manuscrito"] = [" ".join(a.split())[:100] for a in abertos]
'''

CEL_RELATORIO = '''
# SECAO 6 -- relatorio
#
# Escrito a partir de VEREDITOS, nao redigido a mao: o texto nao pode discordar do que
# as celulas mediram. Sai em docs/, junto com o JSON que lhe da lastro.
from datetime import date

ORDEM = [("ambiente", "Ambiente"), ("simbolos", "Simbolos de LaTeX"),
         ("tab7", "Colunas de IC e DM da tabela de variantes"),
         ("calibracao", "Rodada de calibracao canonica"),
         ("semente_do_prophet", "A semente alcanca a banda do Prophet?"),
         ("tabpfn", "TabPFN por janela"), ("assets", "Assets regerados")]

linhas = [f"# Conferencia da regeneracao do XGBoost (REF_ANTES_REGEN_PLACEHOLDER..{REF})",
          "",
          f"Rodado em {date.today():%d/%m/%Y}, em ambiente limpo, a partir do "
          f"commit `{sha}`.", "",
          "Cada veredito abaixo sai de uma medicao deste notebook, "
          "`notebooks/conferencia_regeneracao.ipynb`. O JSON ao lado guarda os numeros.",
          ""]

# Uma linha no topo dizendo se a rodada vale. Um relatorio que enfileira vereditos sem
# dizer que a bancada estava fora do lock convida a citar conclusao que a propria
# rodada nao sustenta.
if VEREDITOS.get("ambiente", {}).get("divergentes"):
    linhas += ["> **Esta rodada nao e conclusiva.** O ambiente nao era o do lock, "
               "entao todo veredito que dependa de valor numerico esta indeterminado. "
               "O que nao depende de execucao -- simbolos e a leitura da tabela de "
               "variantes -- continua valendo.", ""]
for chave, titulo in ORDEM:
    d = VEREDITOS.get(chave)
    if not d:
        continue
    linhas += [f"## {titulo}", "", d.get("conclusao", "sem conclusao registrada"), ""]

if VEREDITOS.get("abertos_no_manuscrito"):
    linhas += ["## Marcas ainda abertas no manuscrito", ""]
    linhas += [f"- {a}" for a in VEREDITOS["abertos_no_manuscrito"]] + [""]

Path("docs").mkdir(exist_ok=True)
Path("docs/conferencia_regeneracao.md").write_text("\\n".join(linhas), encoding="utf-8")
Path("results/conferencia").mkdir(parents=True, exist_ok=True)
Path("results/conferencia/vereditos.json").write_text(
    json.dumps(VEREDITOS, indent=2, ensure_ascii=False, default=str) + "\\n",
    encoding="utf-8")

print("\\n".join(linhas))
print("\\n  docs/conferencia_regeneracao.md")
print("  results/conferencia/vereditos.json")
'''

CEL_COMMIT = '''
# Commit e push numa branch propria. Nada aqui vai para o main direto: o que este
# notebook produz e evidencia, e evidencia entra por revisao como o resto.
BRANCH = "conferencia-regeneracao"  #@param {type:"string"}
ENVIAR = False  #@param {type:"boolean"}

if ENVIAR and not TOKEN:
    print("Sem GITHUB_TOKEN nos Secrets, o push nao tem como autenticar.")
    print("Baixe docs/conferencia_regeneracao.md e results/conferencia/ e commite local.")
elif ENVIAR:
    !git config user.email "$(git log -1 --format=%ae)"
    !git config user.name "$(git log -1 --format=%an)"
    !git checkout -B "$BRANCH"
    !git add docs/conferencia_regeneracao.md results/conferencia paper results
    !git commit -q -m "Confere a regeneracao do XGBoost em ambiente limpo" || echo "nada a commitar"
    !git push -q origin "$BRANCH" && echo "enviado para $BRANCH"
else:
    print("ENVIAR esta desligado. Marque a caixa para commitar e enviar.")
    !git status --short | head -20
'''


def main() -> int:
    if not (ROOT / SERIE_CSV).exists():
        raise SystemExit(f"serie nao encontrada em {SERIE_CSV}")

    celulas = [
        md(f"""
# Conferencia da regeneracao do XGBoost

Confere, em ambiente limpo, o que a regeneracao (`{REF_ANTES_REGEN}..{REF_DEPOIS}`)
mudou e o que ficou em aberto. Cinco perguntas, uma por secao:

1. **Simbolos de LaTeX** -- a regeneracao das figuras trocou simbolos alem do
   `\\blacksquare` ja corrigido? Quais exigem pacote, e o preambulo os carrega?
2. **Tabela de variantes** -- os pontos do XGBoost mudaram e as colunas de IC e DM
   nao. De que par sao essas colunas?
3. **Calibracao** -- a rodada versionada reproduz sob semente fixa? Se sim, ela e a
   canonica.
4. **TabPFN** -- produz a previsao janela a janela e aplica o criterio pre-declarado,
   que ate aqui nao pode ser calculado.
5. **Conferencia geral** -- testes, assets regerados e diff do paper.

As secoes 1 a 3 e 5 rodam sozinhas e levam poucos minutos. A secao 4 e a unica que
precisa de credencial e de tempo (pouco mais de uma hora de API), e fica atras de um
seletor.

A ultima celula escreve `docs/conferencia_regeneracao.md` a partir do que as celulas
mediram, para que o texto nao possa discordar dos numeros.
"""),
        md("## Preparo"),
        code(CEL_SETUP),
        code(CEL_AMBIENTE),
        code(CEL_VERSOES),
        md("## 1. Simbolos de LaTeX"),
        code(CEL_SIMBOLOS),
        code(CEL_SIMBOLOS_PACOTE),
        code(CEL_SIMBOLOS_TESTE),
        md("## 2. Colunas de IC e DM da tabela de variantes"),
        code(CEL_TAB7),
        md("## 3. Rodada de calibracao canonica"),
        code(CEL_CALIBRACAO),
        code(CEL_CALIBRACAO_DIFF),
        code(CEL_CALIBRACAO_SEMENTE),
        md("""## 4. TabPFN por janela

O item em aberto nao e o sMAPE do TabPFN, que ja existe: e a previsao janela a janela.
Sem ela o criterio pre-declarado do artigo nao pode ser calculado, e a afirmacao de que
ele bate as referencias ingenuas fica sem o mesmo lastro que todas as outras do texto.

A margem em jogo e pequena: 0,14 pp sobre o naive sazonal com drift, menor que os
0,177 pp de uma variante que ja reprovou no mesmo teste. O resultado provavel e de
reprovacao, e ele vale tanto quanto uma aprovacao valeria."""),
        code(CEL_TABPFN_MODO),
        code(CEL_TABPFN_TESTE),
        code(CEL_TABPFN_VEREDITO),
        md("## 5. Conferencia geral"),
        code(CEL_CONFERENCIA),
        code(CEL_CONFERENCIA_ASSETS),
        code(CEL_CONFERENCIA_ABERTOS),
        md("## 6. Relatorio"),
        code(CEL_RELATORIO),
        code(CEL_COMMIT),
    ]

    # Os dois valores que o gerador injeta, para nao ficarem digitados no notebook.
    for c in celulas:
        c["source"] = [
            l.replace("SERIE_CSV_PLACEHOLDER", SERIE_CSV)
             .replace("REF_ANTES_REGEN_PLACEHOLDER", REF_ANTES_REGEN)
            for l in c["source"]]

    nb = caderno(celulas)
    valida(nb)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  notebook  {SAIDA.relative_to(ROOT)}  ({len(celulas)} celulas, "
          f"{SAIDA.stat().st_size // 1024} KB)")

    if DRIVE.parent.exists():
        DRIVE.mkdir(parents=True, exist_ok=True)
        copia = DRIVE / SAIDA.name
        copia.write_text(SAIDA.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  copia    {copia}")
    else:
        print("  Drive nao montado: copia nao atualizada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
