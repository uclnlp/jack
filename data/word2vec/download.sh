#!/usr/bin/env bash
cd data/word2vec
wget https://www.dropbox.com/s/bnm0trligffakd9/GoogleNews-vectors-negative300.bin.gz
gunzip GoogleNews-vectors-negative300.bin.gz
cd ../..