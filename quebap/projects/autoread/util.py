import web.embeddings


def init_with_word_embeddings(sess, autoreader,
                              fname="./quebap/data/word2vec/GoogleNews-vectors-negative300.bin",
                              format="word2vec_bin", normalize=True, clean_words=True,):
    """
   Loads embeddings from file and initialize embeddings of autoreader with it

    Parameters
    ----------
    sess: Session (TF)

    autoreader: AutoReader

    fname: string
      Path to file containing embedding

    format: string
      Format of the embedding. Possible values are:
      'word2vec_bin', 'word2vec', 'glove', 'dict'

    normalize: bool, default: True
      If true will normalize all vector to unit length

    clean_words: bool, default: True
      If true will only keep alphanumeric characters and "_", "-"
      Warning: shouldn't be applied to embeddings with non-ascii characters

    load_kwargs:
      Additional parameters passed to load function. Mostly useful for 'glove' format where you
      should pass vocab_size and dim.
    """
    print("Loading autoreader with word2vec embeddings...")
    embeddings = web.embeddings.load_embedding(fname, format, normalize, clean_words=clean_words)

    print("Done loading word2vec!")
    vocab = autoreader.load_vocab()

    em = sess.run(autoreader.input_embeddings)
    for w, j in vocab.items():
        v = embeddings.get(w)
        if v is not None:
            em[j, :v.shape[0]] = v
    sess.run(autoreader.input_embeddings.assign(em))
