import pickle
#from zodbpickle import pickle
import web.embeddings as we
from os import path


def _prepare_embeddings(fname, format, save=True, kwargs={}):
    """
    read embeddings from original file, and store in pickle file
    :param fname: filename
    :param format: word2vec_binary, glove
    :param kwargs: needed in case format=='glove'; dict with keys 'dim' and 'vocab_size'
    """
    fname_out = '.'.join(fname.split('.')[:-1]) + '.pkl'
    emb = we.load_embedding(fname, format, normalize=False, lower=True, clean_words=False, load_kwargs=kwargs)
    if save:
        pickle.dump(emb, open(fname_out, 'wb'))
        print('wrote embeddings to ', fname_out)
    return emb


def load(fname, format='pkl', save=True, kwargs={}):
    """
    load embeddings from prepared pickle file, or from original file
    :param fname: filename
    :param format: pkl | word2vec_binary | glove
    :param kwargs: needed in case format=='glove'; dict with keys 'dim' and 'vocab_size'
    """
    if format == 'pkl':
        # print('load pickle file %s'%fname)
        return pickle.load(open(fname, 'rb'))
    else:
        return _prepare_embeddings(fname, format, save=save, kwargs=kwargs)


if __name__ == '__main__':

    from time import time
    t0 = time()
    #emb_file = 'glove.6B.50d.txt'
    #emb = load(path.join('quebap','data','GloVe',emb_file), 'glove', {'vocab_size':400000,'dim':50})
    #emb = load(path.join('quebap', 'data', 'GloVe', 'glove.6B.50d.pkl'))

    emb_file = 'GoogleNews-vectors-negative300.bin'
    emb = load(path.join('quebap','data','SG_GoogleNews',emb_file),format='word2vec_bin',save=False)
#    emb = load(path.join('quebap','data','SG_GoogleNews',emb_file))

    print('embeddings shape:')
    print(emb.shape)
    print('embedding("hello"):')
    print(emb.get('hello'))
    print('embedding length: %d'%len(emb.get('people')))
    print('total time: %.2fmin'%((time()-t0)/60.))
