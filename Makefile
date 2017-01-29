# simple makefile to simplify repetitive build env management tasks under posix
PYTHON := python3
PIP := pip
PYTEST := pytest

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
test:
	$(PYTEST) tests -v -m "not (overfit or smalldata or smalldatagpu or overfitgpu)"	

# FIXME: this should probably be test-overfit rather than overfit
overfit:
	$(PYTEST) tests -v -m "(overfit and CPU)"
smalldata:
	$(PYTEST) tests -v -m "(smalldata and CPU)"
overfitgpu:
	$(PYTEST) tests -v -m "(overfit and GPU)"
smalldatagpu:
	$(PYTEST) tests -v -m "(smalldata and GPU)"

sentihood:
	$(PYTEST) tests -v -m sentihood
SNLI:
	$(PYTEST) tests -v -m SNLI
doctests:
	$(PYTEST) --doctest-modules jtr/preprocess/vocab.py

gpu:
	$(PYTEST) tests -v -m GPU
cpu:
	$(PYTEST) tests -v -m CPU
