#!/usr/bin/env bash
wget -P data/SNLI/ http://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip data/SNLI/snli_1.0.zip -d data/SNLI
rm data/SNLI/snli_1.0.zip
