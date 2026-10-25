"""Baixa os DBC do SIM das 27 UFs para data/raw/cache (v2, docs/plano_v2_escopo_ampliado.md).

So baixa e confere que cada arquivo decodifica; o painel de series sai de outro script.
Reaproveita o download em camadas de extract_sim_real.py e pula o que ja esta no cache.

Uso: python scripts/v2_baixa_sim_ufs.py --years 2010-2023 [--ufs AC,AL]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_sim_real import decode_dbc, download_via_ftp_direct, download_via_http_mirror, parse_years  # noqa: E402

UFS = ("AC AL AM AP BA CE DF ES GO MA MG MS MT PA PB PE PI PR RJ RN RO RR RS SC SE SP TO").split()
CACHE = Path("data/raw/cache")
MANIFESTO = Path("data/raw/v2_manifesto_sim.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--years", required=True, help="range (2010-2023) ou lista (2019,2020)")
    p.add_argument("--ufs", default=",".join(UFS), help="lista de UFs separada por virgula")
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def baixa(uf: str, ano: int, retries: int) -> Path:
    for tentativa in range(1, retries + 1):
        try:
            return download_via_http_mirror(uf, ano, CACHE)
        except Exception as exc:
            print(f"[WARN] mirror {uf}-{ano} tentativa {tentativa}: {exc}")
            (CACHE / f"DO{uf}{ano}.dbc").unlink(missing_ok=True)
            time.sleep(5 * tentativa)
    print(f"[INFO] {uf}-{ano}: tentando FTP do DataSUS")
    return download_via_ftp_direct(uf, ano, CACHE)


def main() -> int:
    args = parse_args()
    anos = parse_years(args.years)
    ufs = [u.strip().upper() for u in args.ufs.split(",")]
    manifesto = json.loads(MANIFESTO.read_text()) if MANIFESTO.exists() else {}

    falhas = []
    for uf in ufs:
        for ano in anos:
            chave = f"{uf}-{ano}"
            if manifesto.get(chave, {}).get("ok"):
                continue
            try:
                caminho = baixa(uf, ano, args.retries)
                df = decode_dbc(caminho)
                if "CAUSABAS" not in df.columns or len(df) == 0:
                    raise ValueError(f"arquivo sem CAUSABAS ou vazio ({len(df)} linhas)")
                manifesto[chave] = {
                    "ok": True, "arquivo": caminho.name, "linhas": len(df),
                    "mb": round(caminho.stat().st_size / 2**20, 1),
                    "circulatorio": int(df["CAUSABAS"].astype(str).str.startswith("I").sum()),
                }
                print(f"[OK] {chave}: {len(df)} obitos, {manifesto[chave]['circulatorio']} capitulo IX")
            except Exception as exc:
                manifesto[chave] = {"ok": False, "erro": str(exc)[:200]}
                falhas.append(chave)
                print(f"[ERRO] {chave}: {exc}")
            MANIFESTO.write_text(json.dumps(manifesto, indent=1, sort_keys=True))

    print(f"\n{len(ufs) * len(anos) - len(falhas)} de {len(ufs) * len(anos)} arquivos ok; falhas: {falhas or 'nenhuma'}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
