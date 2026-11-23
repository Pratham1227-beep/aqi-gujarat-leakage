# Makefile for AQI Gujarat Leakage Study
#
# Targets:
#   make setup           # create env, install, pre-commit
#   make data            # ingest all sources, write MANIFEST.csv
#   make clean           # validity + aggregate + reconstruct -> processed parquet
#   make protocol-test   # run all leakage and metric tests
#   make tune            # one-time tuning on earliest window
#   make run             # smoke grid on one city
#   make run-all         # full grid
#   make results         # tables from runs.csv
#   make figures         # all figures
#   make paper           # build LaTeX
#   make lint            # ruff + mypy
#   make test            # pytest

.PHONY: setup data clean protocol-test tune run run-all results figures paper lint test

setup:
	uv venv --python 3.11
	uv pip install -e .[dev]
	uv pip install -r requirements.lock
	pre-commit install

data:
	python -m aqi.cli ingest

clean:
	python -m aqi.cli build-daily

protocol-test:
	python -m pytest tests/ -v -k "leak or metric or split or calendar or validity or aggregate or reconstruct"

tune:
	python -m aqi.cli tune

run:
	python -m aqi.cli run --city Ahmedabad --setup A --model all --horizon 1

run-all:
	python -m aqi.cli run-all

results:
	python -m aqi.cli tables

figures:
	python -m aqi.cli figures

paper:
	cd paper && latexmk -pdf main.tex

lint:
	python -m ruff check src tests
	python -m mypy src

test:
	python -m pytest tests -v

# Format code
format:
	python -m ruff format src tests

# Check formatting without applying
check-format:
	python -m ruff format --check src tests