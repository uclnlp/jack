#!/usr/bin/env bash

if [ -f "data/GloVe/glove.840B.300d.txt" ]
then
    echo "data/GloVe/glove.840B.300d.txt already exists! Doing nothing!"
else
    # echo "Downloading glove.840B.300d.txt!"
    # wget -c -P data/GloVe/ http://nlp.stanford.edu/data/glove.840B.300d.zip
    # unzip -d data/GloVe/ data/GloVe/glove.840B.300d.zip
    echo "Downloading glove.840B.300d.memory_map_dir!"
    wget -c -P data/GloVe/ http://data.neuralnoise.com/jack/embeddings/glove.840B.300d.memory_map_dir.tar.gz
    tar xvfz data/GloVe/glove.840B.300d.memory_map_dir.tar.gz -C data/GloVe/
fi
