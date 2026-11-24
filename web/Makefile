.PHONY: dev build test

ifeq ($(OS),Windows_NT)
PYTHON=.venv312/Scripts/python.exe
NPM=npm.cmd
else
PYTHON=.venv312/bin/python
NPM=npm
endif

dev:
	$(PYTHON) scripts/dev.py

build:
	cd frontend && $(NPM) run build

test:
	$(PYTHON) -m pytest reference/tests tests -q
