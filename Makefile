PYTHON ?= python3
export PYTHONPATH := src

.PHONY: help data factors portfolios bias evaluate figures all test lint clean clean-data

help:
	@echo "data        download and clean every dataset (slow on a cold cache)"
	@echo "factors     weekly exposures, factor returns, survivorship gap"
	@echo "portfolios  fit the risk models, run the USMV and SPLV rules"
	@echo "bias        weekly bias statistics for all four estimators"
	@echo "evaluate    replication accuracy, horse race, risk and alpha tables"
	@echo "figures     the charts the report embeds"
	@echo "all         the whole pipeline in order"
	@echo "test        run the test suite"
	@echo "clean       remove generated reports and caches, keep raw downloads"
	@echo "clean-data  also remove raw downloads (forces a full refetch)"

data:
	$(PYTHON) scripts/01_build_data.py

factors:
	$(PYTHON) scripts/02_build_factors.py

portfolios:
	$(PYTHON) scripts/03_portfolios.py

bias:
	$(PYTHON) scripts/04_bias_tests.py

evaluate:
	$(PYTHON) scripts/05_evaluate.py

figures:
	$(PYTHON) scripts/06_figures.py

# Order matters: each stage reads what the previous one wrote.
all: data factors portfolios bias evaluate figures

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m pyflakes src scripts tests || true

clean:
	rm -rf data/interim data/processed reports/figures reports/tables
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +

clean-data: clean
	rm -rf data/raw
