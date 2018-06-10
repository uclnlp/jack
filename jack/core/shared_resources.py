"""Shared resources are used to store reader all stateful information about a reader and share it between modules.

Examples are include the vocabulary, hyper-parameters or name of a reader that are mostly stored in a configuration
dict. Shared resources are also used later to setup an already saved reader.
"""

import os
import pickle

import yaml

from jack.io.embeddings import Embeddings
from jack.util.vocab import Vocab


class SharedResources:
    """Shared resources between modules.

    A class to provide and store generally shared resources, such as vocabularies,
    across the reader sub-modules.
    """

    def __init__(self, vocab: Vocab = None, config: dict = None, embeddings: Embeddings = None):
        """
        Several shared resources are initialised here, even if no arguments
        are passed when calling __init__.
        The instantiated objects will be filled by the InputModule.
        - self.config holds hyperparameter values and general configuration
            parameters.
        - self.vocab serves as default Vocabulary object.
        - self.answer_vocab is by default the same as self.vocab. However,
            this attribute can be changed by the InputModule, e.g. by setting
            sepvocab=True when calling the setup_from_data() of the InputModule.
        """
        self.config = config or dict()
        self.vocab = vocab
        self.embeddings = embeddings

    def store(self, path):
        """
        Saves all attributes of this object.

        Args:
            path: path to save shared resources
        """
        if not os.path.exists(path):
            os.mkdir(path)
        vocabs = [(k, v) for k, v in self.__dict__.items() if isinstance(v, Vocab)]
        with open(os.path.join(path, 'remainder'), 'wb') as f:
            remaining = {k: v for k, v in self.__dict__.items()
                         if not isinstance(v, Vocab) and not k == 'config' and not k == 'embeddings'}
            pickle.dump(remaining, f, pickle.HIGHEST_PROTOCOL)
        for k, v in vocabs:
            v.store(os.path.join(path, k))
        with open(os.path.join(path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        if self.embeddings is not None:
            self.embeddings.store(os.path.join(path, 'embeddings'))

    def load(self, path):
        """
        Loads this (potentially empty) resource from path (all object attributes).
        Args:
            path: path to shared resources
        """
        remainder_path = os.path.join(path, 'remainder')
        if os.path.exists(remainder_path):
            with open(remainder_path, 'rb') as f:
                self.__dict__.update(pickle.load(f))
        for f in os.listdir(path):
            if f == 'config.yaml':
                with open(os.path.join(path, f), 'r') as f:
                    self.config = yaml.load(f)
            elif f == 'embeddings':
                self.embeddings = Embeddings.from_dir(os.path.join(path, f))
            else:
                v = Vocab()
                v.load(os.path.join(path, f))
                self.__dict__[f] = v
