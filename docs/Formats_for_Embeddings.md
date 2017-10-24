# Formats for Embeddings

Jack supports loading of various embedding formats, including glove and word2vec. These can be specified in the 
configuration files or command line parameters of your models via the `embedding_format` parameter. In particular,
we support

* `glove`: the original glove format, either as txt file or zipped
* `word2vec`: word2vec format
* `fasttext`: fasttext format
* `memory_map_dir`: a directory that contains the embeddings as a numpy memory map, and meta information necessary to
instantiate it. 

## Memory Map Directories
For large embeddings (large dimensions, many words), loading embeddings into memory can both take up a lot of 
CPU memory, and be very slow. Numpy provides a file format for matrices that loads vectors on the fly. In Jack
this functionality is used via the `memory_map_dir` format. 

You can convert your embeddings into this format via the `memory_map.py` script. For example, to convert Glove embeddings,
assuming you are in the top level jack directory, write:

```bash
$ export PYTHONPATH=$PYTHONPATH:.
$ python3 bin/mmap-cli.py --help
$ python3 bin/mmap-cli.py data/GloVe/glove.840B.300d.txt data/GloVe/glove.840B.300d.memory_map_dir
```

This creates a directory `data/GloVe/glove.840B.300d.memory_map_dir` that stores the memory map and some necessary
meta information.

Using this format can substantially reduce start-up times and memory footprint.
