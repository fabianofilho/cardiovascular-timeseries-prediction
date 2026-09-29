# Todos os alvos usam o interpretador do .venv DIRETAMENTE, sem `activate`.
#
# Antes o Makefile fazia `. .venv/bin/activate && ...`, que nao existe no Windows: o venv
# poe o interpretador em .venv/Scripts/. Na pratica os alvos rodavam no Python do sistema,
# que e como os 239 testes acabaram rodando fora do ambiente declarado. Chamar o
# interpretador pelo caminho dispensa ativacao e funciona igual nos dois sistemas.

VENV ?= .venv

ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV)/Scripts/python.exe
else
VENV_PY := $(VENV)/bin/python
endif

PYTHON ?= python
PIP = $(VENV_PY) -m pip
RUN = PYTHONPATH=src $(VENV_PY)

.PHONY: venv setup-base setup-full lock check-venv sample-pysus smoke-baseline \
        benchmark-all assets test regen-calendario

# Falha cedo e com instrucao, em vez de cair num "No such file" do make.
check-venv:
	@test -x "$(VENV_PY)" || { \
		echo "ambiente ausente em $(VENV). Rode: make setup-full"; exit 1; }

venv:
	$(PYTHON) -m venv $(VENV)

setup-base: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-full: setup-base
	$(PIP) install -r requirements-optional.txt

# Congela o que ESTA instalado, que e diferente do que os requirements PEDEM: eles trazem
# faixas (>=) e o lock traz a versao exata que produziu os numeros. Sem isso, "instalei os
# requirements" nao identifica um ambiente.
lock: check-venv
	$(PIP) freeze --exclude-editable > requirements-lock.txt
	@echo "requirements-lock.txt atualizado"

sample-pysus: check-venv
	$(RUN) scripts/prepare_pysus_sample.py \
		--source sih \
		--uf SP \
		--year 2024 \
		--month 1 \
		--sample-rows 5000 \
		--raw-output data/raw/pysus_sih_sp_2024_01_sample.csv \
		--series-output data/processed/serie_eventos_sp_sample.csv

smoke-baseline: check-venv
	$(RUN) scripts/run_benchmark.py \
		--input-csv data/processed/serie_eventos_sp_tiny.csv \
		--date-col date \
		--value-col value \
		--freq MS \
		--horizon 3 \
		--min-train-size 12 \
		--models sarima \
		--output-prefix results/benchmark_sp_tiny

benchmark-all: check-venv
	$(RUN) scripts/run_benchmark.py \
		--input-csv data/processed/serie_eventos_sp_sample.csv \
		--date-col date \
		--value-col value \
		--freq MS \
		--horizon 6 \
		--min-train-size 24 \
		--models sarima,prophet,timesfm \
		--output-prefix results/benchmark_sp_sample_full

assets: check-venv
	$(RUN) scripts/build_paper_assets.py

test: check-venv
	$(RUN) -m pytest

# Regenera a tabela e a figura do termo de calendario (revisao_*_temp_diffcal). Rodar SO na
# maquina que gerou os numeros publicados: o controle confere que a linha "base" reproduz a
# Tabela 5 e para tudo se nao reproduzir. Ver docs/xgboost_reprodutibilidade.md.
regen-calendario: check-venv
	$(RUN) scripts/revisao/run_temp_melhorado.py
	$(RUN) scripts/revisao/check_controle_calendario.py
	$(RUN) scripts/revisao/build_revisao_tabs.py
	$(RUN) scripts/revisao/build_revisao_figs2.py
