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
	$(PYTEST) tests -v -m "not (test-overfit or test-smalldata or test-smalldatagpu or test-overfitgpu)"	
test-overfit:
	$(PYTEST) tests -v -m "(test-overfit and test-CPU)"
test-smalldata:
	$(PYTEST) tests -v -m "(test-smalldata and test-CPU)"
test-overfitgpu:
	$(PYTEST) tests -v -m "(test-overfit and test-GPU)"
test-smalldatagpu:
	$(PYTEST) tests -v -m "(test-smalldata and test-GPU)"

test-sentihood:
	$(PYTEST) tests -v -m test-sentihood
test-SNLI:
	$(PYTEST) tests -v -m test-SNLI
doctests:
	$(PYTEST) --doctest-modules jtr/preprocess/vocab.py
