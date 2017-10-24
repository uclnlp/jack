#!/usr/bin/env bash
wget -P data/MultiNLI/ https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip data/MultiNLI/multinli_1.0.zip -d data/MultiNLI
rm data/MultiNLI/multinli_1.0.zip
