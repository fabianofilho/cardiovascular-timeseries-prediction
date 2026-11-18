"""Gera o .docx da v1b para coautores (pandoc, bibliografia por citeproc, figuras em PNG).

Rodar de dentro de paper_v1b/:

    python build_docx.py            -> ../build_v1b/manuscript_v1b.docx

O .tex passa por um pre-processamento antes do pandoc: tira o preambulo e o ramo cego,
troca as figuras pgfplots pelo PNG gerado junto e transforma \\aberto e \\missing em texto
marcado, para o coautor ver o buraco no Word.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SAIDA = Path("../build_v1b/manuscript_v1b.docx")


def preprocessar(tex: str) -> str:
    tex = tex.replace("\\input{../paper/preamble}", "")
    # \ifdefined\BLIND A \else B \fi -> B ; \ifdefined\BLIND A \fi -> nada
    tex = re.sub(r"\\ifdefined\\BLIND(.*?)\\else(.*?)\\fi", r"\2", tex, flags=re.S)
    tex = re.sub(r"\\ifdefined\\BLIND.*?\\fi", "", tex, flags=re.S)
    tex = re.sub(r"\\input\{\.\./paper/figures/(\w+)\}",
                 r"\\includegraphics[width=\\textwidth]{../paper/figures/\1.png}", tex)
    tex = re.sub(r"\\(aberto|missing)\{([^}]*)\}", r"\\textbf{[OPEN: \2]}", tex)
    tex = tex.replace("\\input{supplement}", Path("supplement.tex").read_text(encoding="utf-8"))
    tex = re.sub(r"\\input\{\.\./paper/figures/(\w+)\}",
                 r"\\includegraphics[width=\\textwidth]{../paper/figures/\1.png}", tex)
    return tex


def main() -> int:
    tex = preprocessar(Path("manuscript.tex").read_text(encoding="utf-8"))
    tmp = Path("../build_v1b/_docx_src.tex")
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(tex, encoding="utf-8")
    cmd = ["pandoc", str(tmp), "-o", str(SAIDA), "--from=latex", "--citeproc",
           "--bibliography=refs/references.bib", "--resource-path=.:..:../paper",
           "--number-sections"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stderr.strip()[:2000])
    print(f"{SAIDA} ({SAIDA.stat().st_size // 1024} KB)" if SAIDA.exists() else "falhou")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
