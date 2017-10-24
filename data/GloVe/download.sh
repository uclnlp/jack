#!/usr/bin/env bash

if [ -f "data/GloVe/glove.840B.300d.txt" ]
then
    echo "data/GloVe/glove.840B.300d.txt already exists! Doing nothing!"
else
    echo "Downloading glove.840B.300d.txt!"
    wget -P data/GloVe/ http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip data/GloVe/glove.840B.300d.zip
fi
