import logging
import pickle
import numpy as np
import sys

from jtr.io.embeddings import Embeddings, load_embeddings


def load_memory_map(file_prefix: str) -> Embeddings:
    """
    Loads embeddings from a memory map file to allow lazy loading (and reduce the memory usage).
    Args:
        file_prefix: a file prefix. This function stores several files, and they will all start with this prefix.

    Returns:
        Embeddings object with a lookup matrix that is backed by a memory map.

    """
    meta_file = file_prefix + "_meta.pkl"
    mem_map_file = file_prefix + "_memmap"
    with open(meta_file, "rb") as f:
        meta = pickle.load(f)
    shape = meta['shape']
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='r+', shape=shape)
    result = Embeddings(meta['vocab'], mem_map, filename=file_prefix, emb_format="mem_map")
    return result


def save_as_memory_map(file_prefix: str, emb: Embeddings):
    meta_file = file_prefix + "_meta.pkl"
    mem_map_file = file_prefix + "_memmap"
    with open(meta_file, "wb") as f:
        pickle.dump({
            "vocab": emb.vocabulary,
            "shape": emb.shape
        }, f)
    mem_map = np.memmap(mem_map_file, dtype='float32', mode='w+', shape=emb.shape)
    mem_map[:] = emb.lookup[:]
    mem_map.flush()
    del mem_map


if __name__ == "__main__":
    input_name = sys.argv[1]
    output_prefix = sys.argv[2]
    embeddings = load_embeddings(input_name)
    logging.info("Loaded embeddings from {}".format(input_name))
    save_as_memory_map(output_prefix, embeddings)
    logging.info("Stored embeddings to {}".format(output_prefix))

