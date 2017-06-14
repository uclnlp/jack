# simple makefile to simplify repetitive build env management tasks under posix
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest

init:
	$(PIP) install -r requirements.txt
install:
	$(PYTHON) setup.py install
install-develop:
	$(PYTHON) setup.py develop
install-user:
	$(PYTHON) setup.py install --user
clean:
	$(PYTHON) setup.py clean --all
unittest:
	$(PYTEST) tests -v -m "not (overfit or smalldata)" -k "not test_pipeline"	
test:
	$(PYTEST) tests -v -m "not (smalldata)"

# FIXME: this should probably be test-overfit rather than overfit
overfit:
	$(PYTEST) tests -v -m "overfit"
smalldata:
	$(PYTEST) tests -v -m "smalldata"

SNLI:
	$(PYTEST) tests -v -m SNLI
doctests:
	$(PYTEST) --doctest-modules jtr/preprocess/vocab.py
