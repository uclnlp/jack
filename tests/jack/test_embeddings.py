from jack.io.embeddings import load_embeddings
import numpy as np


def test_memory_map_dir():
    import tempfile
    from jack.io.embeddings.memory_map import save_as_memory_map_dir, load_memory_map_dir
    embeddings_file = "data/GloVe/glove.the.50d.txt"
    embeddings = load_embeddings(embeddings_file, 'glove')
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem_map_dir = tmp_dir + "/glove.the.50d.memmap"
        save_as_memory_map_dir(mem_map_dir, embeddings)
        loaded_embeddings = load_memory_map_dir(mem_map_dir)
        assert loaded_embeddings.shape == embeddings.shape
        assert len(loaded_embeddings.vocabulary) == 1
        assert loaded_embeddings.vocabulary["the"] == 0
        assert "foo" not in loaded_embeddings.vocabulary
        assert np.isclose(loaded_embeddings.get("the"), embeddings.get("the"), 1.e-5).all()
