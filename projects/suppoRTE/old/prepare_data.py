"""
based on Dirk Weissenborn's train_snli, modified to load jtr data files
"""
from os import path
import json
from jtr.io.SNLI2jtr_v2 import CONJ
import nltk

def _load_snli_jtr_v2(fname):
    instances = json.load(open(fname,'r'))['instances']
    A,B,S = [],[],[]
    for instance in instances:
        sA,sB = instance['questions'][0]['question'].split(CONJ)
        A.append(sA)
        B.append(sB)
        S.append(instance['questions'][0]['answers'][0]['text'])
    return A,B,S


def load_data(loc, embeddings):
    """
    Load the SNLI dataset (jtr encoding version 2)
    :param loc: folder with jtr data files
    :param embeddings: embeddings as web.embedding object 
    """

    devA,devB,devS = _load_snli_jtr_v2(path.join(loc, 'snli_1.0_dev_jtr_v2.json'))
    trainA,trainB,trainS = _load_snli_jtr_v2(path.join(loc, 'snli_1.0_train_jtr_v2.json'))
    testA,testB,testS = _load_snli_jtr_v2(path.join(loc, 'snli_1.0_test_jtr_v2.json'))

    vocab, oo_vocab = dict(), dict()
    def encode(sentence):
        """
        encode sentence, while assigning and saving id's to newly encountered words (in vocab or oo_vocab, dep. on embeddings)
        """
        if "<s>" not in oo_vocab:
            oo_vocab["<s>"] = len(oo_vocab)
        if "</s>" not in oo_vocab:
            oo_vocab["</s>"] = len(oo_vocab)
        word_ids = [-oo_vocab["<s>"]-1]
        for w in nltk.word_tokenize(sentence.lower()):
            #out-of-embeddings-vocab: negative word_ids; in-vocab: word_ids >=0
            wv = embeddings.get(w)
            if wv is None:
                if w not in oo_vocab:
                    oo_vocab[w] = len(oo_vocab)
                word_ids.append(-oo_vocab[w]-1)
            else:
                if w not in vocab:
                    vocab[w] = len(vocab)
                word_ids.append(vocab[w])
        word_ids.append(-oo_vocab["</s>"]-1)
        return word_ids

    #encode all data while constructing vocab and oo_vocab (map word symbols to ids)
    devA = [encode(s) for s in devA]
    devB = [encode(s) for s in devB]
    trainA = [encode(s) for s in trainA]
    trainB = [encode(s) for s in trainB]
    testA = [encode(s) for s in testA]
    testB = [encode(s) for s in testB]

    def _normalize_ids(ds):#ds: dataset
        """
        after all oo_vocab words are stored, shift all vocab ids for dataset ds up by len(oo_vocab); 
        within-vocab id's start at len(oo_vocab); oo_vocab id's are from 0 to len(oo_vocab)-1
        (oo_vocab id -n => N-n, hence requires correction below)
        """
        for word_ids in ds:
            for i in range(len(word_ids)):
                word_ids[i] += len(oo_vocab)

    _normalize_ids(trainA)
    _normalize_ids(trainB)
    _normalize_ids(devA)
    _normalize_ids(devB)
    _normalize_ids(testA)
    _normalize_ids(testB)

    #make sure oo_vocab words are mapped to correct (shifted) positive ids  
    for k in oo_vocab:
        oo_vocab[k] = len(oo_vocab) - oo_vocab[k] - 1

    return trainA, trainB, devA, devB, testA, testB, [trainS, devS, testS], vocab, oo_vocab



if __name__=="__main__":
    import jtr.projects.suppoRTE.prepare_embeddings as emb 
    from time import time
    
    #load embeddings for testing    
    t0 = time()
    #embeddings = emb.load(path.join('jtr','data','GloVe','glove.6B.50d.txt'), 'glove', {'vocab_size':400000,'dim':50})
    emb_file = 'glove.6B.50d.pkl'
    embeddings = emb.load(path.join('jtr','data','GloVe',emb_file))
    print('loaded %s embeddings in %.0fs'%(emb_file,time()-t0))
    
    #encode SNLI data with embeddings   
    t1 = time()
    SNLIfolder = path.join('jtr','data','SNLI','snli_1.0')
    trainA, trainB, devA, devB, testA, testB, labels, vocab, oo_vocab = load_data(SNLIfolder,embeddings)
    print('loaded SNLI data in %.0fs'%(time()-t1))
    print('found %d terms with embeddings, and %d without.'%(len(vocab),len(oo_vocab)))
