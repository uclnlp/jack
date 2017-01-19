#!/usr/bin/env bash
wget -P jtr/data/SNLI/ http://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip jtr/data/SNLI/snli_1.0.zip -d jtr/data/SNLI
rm jtr/data/SNLI/snli_1.0.zip
