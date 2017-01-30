from jtr.load.embeddings.word_to_vec import load_word2vec
from jtr.load.embeddings.glove import load_glove
import zipfile


class Embeddings:
    """Wraps Vocabulary and embedding matrix to do lookups"""
    def __init__(self, vocabulary, lookup):
        self.vocabulary = vocabulary
        self.lookup = lookup

    def get(self, word):
        id = self.vocabulary.get_idx_by_word(word)
        #out-of-vocab word
        if id is None:
            return None  #lookup[None] would return entire lookup table
        #known embedding
        else:
            return self.lookup[id]

    @property
    def shape(self):
        return self.lookup.shape


def load_embeddings(file, typ, **options):
    """Loads either GloVe or word2vec embeddings and wraps it into Embeddings
    Args:
        file (string): Path to files like "GoogleNews-vectors-negative300.bin.gz"
           or "glove.42B.300d.zip"
        typ: (string(: Either "word2vec" or "glove"
    Returns:
        Embeddings: Wrapper class around Vocabulary embedding matrix.
    """
    assert typ in ["word2vec", "glove"], "so far only 'word2vec' and 'glove' foreseen"

    if typ.lower() == "word2vec":
        return Embeddings(*load_word2vec(file, **options))

    elif typ.lower() == "glove":
        if file.endswith('.txt'):
            with open(file, 'rb') as f:
                return Embeddings(*load_glove(f))
        elif file.endswith('.zip'):
            #for files glove.840B.300d.zip (not glove.6B.zip)
            with zipfile.ZipFile(file) as zf:
                txtfile = file.split('/')[-1][:-4]+'.txt'
                with zf.open(txtfile,'r') as f:
                    return Embeddings(*load_glove(f))
        else:
            raise NotImplementedError
