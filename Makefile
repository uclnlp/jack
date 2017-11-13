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


# ===========
# SNLI
# ===========

snli:
	./data/SNLI/download.sh
snli-esim:
	 $(PYTHON) bin/jack-train.py with \
		loader=snli \
		train=data/SNLI/snli_1.0/snli_1.0_train.jsonl \
		dev=data/SNLI/snli_1.0/snli_1.0_dev.jsonl \
		test=data/SNLI/snli_1.0/snli_1.0_test.jsonl \
		model=esim_snli_reader


# ===========
# SQuAD
# ===========
#
squad:
	./data/SQuAD/download.sh

glove:
	./data/GloVe/download.sh

squad-fastqa:
	#$(PYTHON) bin/jack-train.py with config='./conf/fastqa.yaml'
	$(PYTHON) bin/jack-train.py with \
		train=data/SQuAD/train-v1.1.json \
		dev=data/SQuAD/dev-v1.1.json \
		model=fastqa_reader \
		loader=squad



doctests:
	$(PYTEST) --doctest-modules jtr/preprocess/vocab.py

help:
	@echo "=================================="
	@echo "Download and preprocess datasets:"
	@echo "=================================="
	@echo ""
	@echo "SNLI"
	@echo ""
	@echo "=================================="
	@echo "Run models:"
	@echo "=================================="
	@echo ""
	@echo "SNLI-esim"


