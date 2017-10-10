#!/usr/bin/env bash

cd "$(dirname "$0")"

if [ -f "glove.6B.50d.txt" ]
then
    echo "glove.6B.50d.txt already exists! Doing nothing!"
else
    echo "Downloading glove.6B.50d.txt!"
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
fi
