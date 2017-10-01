from jtr.io.embeddings import load_embeddings
import numpy as np


def test_memory_maps():
    import tempfile
    from jtr.io.embeddings.embeddings import save_as_memory_map, load_memory_map
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    embeddings = load_embeddings(embeddings_file, 'glove')
    with tempfile.TemporaryDirectory() as tmp_dir:
        prefix = tmp_dir + "_memmap_emb"
        save_as_memory_map(prefix, embeddings)
        loaded_embeddings = load_memory_map(prefix)
        assert loaded_embeddings.shape == embeddings.shape
        assert len(loaded_embeddings.vocabulary) == 1
        assert loaded_embeddings.vocabulary.get_idx_by_word(b"the") == 0
        assert loaded_embeddings.vocabulary.get_idx_by_word(b"foo") is None
        assert np.isclose(loaded_embeddings.get(b"the"), embeddings.get(b"the"), 1.e-5).all()
