"""Incerteza do benchmark: bootstrap por bloco de janelas e Diebold-Mariano.

Por que estas funcoes moram no pacote e nao no script: todo intervalo de confianca do
manuscrito sai daqui. Um erro aqui nao produz excecao nem valor absurdo, produz um
intervalo mais estreito do que deveria, e intervalo estreito e exatamente o que faz uma
diferenca de ruido parecer achado. Foi assim que a primeira rodada deste projeto quase
publicou um ranking que os dados nao sustentavam.

A unidade de reamostragem e a JANELA INTEIRA, com seus horizontes juntos. As previsoes
de um rolling origin nao sao independentes: as janelas se sobrepoem e, dentro de cada
uma, os seis horizontes compartilham a mesma origem de treino. Reamostrar previsao a
previsao trataria 618 observacoes dependentes como 618 independentes e daria um IC
falsamente estreito.

Ver tests/test_uncertainty.py.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sortear_janelas(n_win: int, b: int, seed: int) -> np.ndarray:
    """Matriz (b, n_win) de indices de janela reamostrados com reposicao.

    Sorteada UMA vez e reusada por todos os modelos e todas as metricas. E isso que
    torna o bootstrap pareado: em cada replica, todos os modelos sao avaliados sobre
    exatamente o mesmo conjunto de janelas, entao a diferenca entre eles nao carrega
    variacao de amostragem propria.
    """
    return np.random.default_rng(seed).integers(0, n_win, size=(b, n_win))


def bootstrap_metrica(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Distribuicao de replicas da media de `mat`, reamostrando linhas (janelas).

    `mat` tem forma (n_win, n_hor): a media e sobre os dois eixos, entao a janela
    inteira entra ou sai junto, com todos os seus horizontes.
    """
    return mat[idx].mean(axis=(1, 2))


def ic_percentil(boot: np.ndarray, nivel: float = 95.0) -> tuple[float, float]:
    """IC pelo metodo do percentil."""
    cauda = (100.0 - nivel) / 2.0
    lo, hi = np.percentile(boot, [cauda, 100.0 - cauda])
    return float(lo), float(hi)


def p_bicaudal(boot_dif: np.ndarray) -> float:
    """p-valor bicaudal do bootstrap: fracao de replicas do outro lado do zero.

    Limitado a 1.0 porque, com a diferenca centrada em zero, as duas fracoes podem
    somar mais que 1 e dobrar a menor delas passaria de 1.
    """
    p = 2.0 * min((boot_dif >= 0).mean(), (boot_dif <= 0).mean())
    return float(min(p, 1.0))


def dm_test(d: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano com correcao de Harvey, Leybourne e Newbold.

    `d` e a serie do diferencial de perda por janela, para um horizonte fixo. A
    variancia de longo prazo usa truncamento em h-1 defasagens, que e o padrao para
    previsao h passos a frente. Sem a correcao de Harvey o teste rejeita demais em
    amostra pequena, e 103 janelas e amostra pequena para este fim.
    """
    n = len(d)
    d_bar = d.mean()
    gamma0 = np.sum((d - d_bar) ** 2) / n
    var = gamma0
    for lag in range(1, h):
        cov = np.sum((d[lag:] - d_bar) * (d[:-lag] - d_bar)) / n
        var += 2.0 * cov
    if var <= 0:  # variancia de longo prazo negativa: truncamento nao confiavel
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(var / n)
    correcao = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrigido = dm * correcao
    p = 2 * (1 - stats.t.cdf(abs(dm_corrigido), df=n - 1))
    return float(dm_corrigido), float(p)
