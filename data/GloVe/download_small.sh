#!/usr/bin/env bash

if [ -f "data/GloVe/glove.6B.50d.txt" ]
then
    echo "glove.6B.50d.txt already exists! Doing nothing!"
else
    echo "Downloading glove.6B.50d.txt!"
    wget -c -P data/GloVe/ http://nlp.stanford.edu/data/glove.6B.zip
    unzip data/GloVe/glove.6B.zip -d data/GloVe
fi
