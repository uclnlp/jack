#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from jack.io.embeddings import load_embeddings
from jack.io.embeddings.memory_map import save_as_memory_map_dir

import logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert embeddings to memory map directory')
    parser.add_argument("input_file", help="The input embedding file.")
    parser.add_argument("output_dir",
                        help="The name of the directory to store the memory map in. Will be created if it doesn't "
                             "exist.")
    parser.add_argument("-f", "--input_format", help="Format of input embeddings.", default="glove",
                        choices=["glove", "word2vec", "memory_map_dir"])
    args = parser.parse_args()
    input_name = args.input_file
    output_dir = args.output_dir
    embeddings = load_embeddings(input_name, typ=args.input_format)
    logging.info("Loaded embeddings from {}".format(input_name))
    save_as_memory_map_dir(output_dir, embeddings)
    logging.info("Stored embeddings to {}".format(output_dir))


if __name__ == "__main__":
    main()
