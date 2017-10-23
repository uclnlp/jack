#!/usr/bin/env bash


if [ -f "data/SQuAD/dev-v1.1.json" ]
then
    echo "Already downloaded."
else
    wget -P data/SQuAD/ https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
    wget -P data/SQuAD/ https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
fi
