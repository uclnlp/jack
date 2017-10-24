#!/usr/bin/env bash
wget -P data/MultiNLI/ https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip data/MultiNLI/multinli_1.0.zip -d data/MultiNLI
rm data/MultiNLI/multinli_1.0.zip

# Create joint dev set
cat data/MultiNLI/multinli_1.0/multinli_1.0_dev_matched.jsonl data/MultiNLI/multinli_1.0/multinli_1.0_dev_mismatched.jsonl > data/MultiNLI/multinli_1.0/multinli_1.0_dev.jsonl
