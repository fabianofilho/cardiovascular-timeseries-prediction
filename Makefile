PYTHON ?= python3
PIP ?= pip
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: venv setup-base setup-full sample-pysus smoke-baseline benchmark-all

venv:
	$(PYTHON) -m venv $(VENV)

setup-base: venv
	$(ACTIVATE) && $(PIP) install --upgrade pip
	$(ACTIVATE) && $(PIP) install -r requirements.txt

setup-full: setup-base
	$(ACTIVATE) && $(PIP) install -r requirements-optional.txt

sample-pysus:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) scripts/prepare_pysus_sample.py \
		--source sih \
		--uf SP \
		--year 2024 \
		--month 1 \
		--sample-rows 5000 \
		--raw-output data/raw/pysus_sih_sp_2024_01_sample.csv \
		--series-output data/processed/serie_eventos_sp_sample.csv

smoke-baseline:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) scripts/run_benchmark.py \
		--input-csv data/processed/serie_eventos_sp_tiny.csv \
		--date-col date \
		--value-col value \
		--freq MS \
		--horizon 3 \
		--min-train-size 12 \
		--models sarima \
		--output-prefix results/benchmark_sp_tiny

benchmark-all:
	$(ACTIVATE) && PYTHONPATH=src $(PYTHON) scripts/run_benchmark.py \
		--input-csv data/processed/serie_eventos_sp_sample.csv \
		--date-col date \
		--value-col value \
		--freq MS \
		--horizon 6 \
		--min-train-size 24 \
		--models sarima,prophet,timesfm \
		--output-prefix results/benchmark_sp_sample_full
