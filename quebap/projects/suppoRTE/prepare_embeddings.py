import pickle
import web.embeddings as we
from os import path


def _prepare_embeddings(fname, format, kwargs={}):
    """
    read embeddings from original file, and store in pickle file
    :param fname: filename
    :param format: word2vec_binary, glove 
    :param kwargs: needed in case format=='glove'; dict with keys 'dim' and 'vocab_size'
    """
    fname_out = '.'.join(fname.split('.')[:-1]) + '.pkl'
    emb = we.load_embedding(fname, format, normalize=False, lower=True, clean_words=False, load_kwargs=kwargs)    
    pickle.dump(emb,open(fname_out,'wb'))
    print('wrote embeddings to ',fname_out)
    return emb

def load(fname, format='pkl', kwargs={}):
    """
    load embeddings from prepared pickle file, or from original file
    :param fname: filename
    :param format: pkl | word2vec_binary | glove
    :param kwargs: needed in case format=='glove'; dict with keys 'dim' and 'vocab_size'
    """
    if format=='pkl':
        #print('load pickle file %s'%fname)
        return pickle.load(open(fname,'rb'))
    else:
        return _prepare_embeddings(fname, format, kwargs)



if __name__ == '__main__':
    #emb = load(path.join('quebap','data','GloVe','glove.6B.50d.txt'), 'glove', {'vocab_size':400000,'dim':50})
    emb = load(path.join('quebap','data','GloVe','glove.6B.50d.pkl'))
    print(emb.shape)
    print(emb.get('hello'))
    
    