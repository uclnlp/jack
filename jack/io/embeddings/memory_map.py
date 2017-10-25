import json
import logging
import os

import numpy as np

from jack.io.embeddings import Embeddings, load_embeddings


def load_memory_map_dir(directory: str) -> Embeddings:
    """
    Loads embeddings from a memory map directory to allow lazy loading (and reduce the memory usage).
    Args:
        directory: a file prefix. This function loads two files in the directory: a meta json file with shape information
        and the vocabulary, and the actual memory map file.

    Returns:
        Embeddings object with a lookup matrix that is backed by a memory map.

    """
    meta_file = os.path.join(directory, "meta.json")
    mem_map_file = os.path.join(directory, "memory_map")
    with open(meta_file, "r") as f:
        meta = json.load(f)
    shape = tuple(meta['shape'])
    vocab = meta['vocab']
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='r+', shape=shape)
    result = Embeddings(vocab, mem_map, filename=directory, emb_format="memory_map_dir")
    return result


def save_as_memory_map_dir(directory: str, emb: Embeddings):
    """
    Saves the given embeddings as memory map file and corresponding meta data in a directory.
    Args:
        directory: the directory to store the memory map file in (called `memory_map`) and the meta file (called
        `meta.json` that stores the shape of the memory map and the actual vocabulary.
        emb: the embeddings to store.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    meta_file = os.path.join(directory, "meta.json")
    mem_map_file = os.path.join(directory, "memory_map")
    with open(meta_file, "w") as f:
        json.dump({
            "vocab": emb.vocabulary,
            "shape": emb.shape
        }, f)
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='w+', shape=emb.shape)
    mem_map[:] = emb.lookup[:]
    mem_map.flush()
    del mem_map


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
