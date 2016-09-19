class Vocab(object):
    def __init__(self, unk="<UNK>"):
        self.sym2id = {unk: 0}
        self.id2sym = [unk]
        self.sym2freqs = {}
        self.unk = unk
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get_id(self, sym):
        if not self.frozen:
            if sym not in self.sym2id:
                self.sym2id[sym] = len(self.id2sym)
                self.id2sym.append(sym)
                self.sym2freqs[sym] = 1
            else:
                self.sym2freqs[sym] += 1
        if sym in self.sym2id:
            return self.sym2id[sym]
        else:
            return self.sym2id[self.unk]

    def get_sym(self, id):
        return self.id2sym[id]

    def __call__(self, *args, **kwargs):
        symbols = args
        if len(args) == 1:
            if isinstance(args[0], list):
                symbols = args[0]
            else:
                return self.get_id(args[0])

        return [self.get_id(sym) for sym in symbols]

    def __len__(self):
        return len(self.id2sym)


if __name__ == '__main__':
    vocab = Vocab()
    print(vocab("blah"))
    print(vocab("bluh"))
    print(vocab("bleh"))
    print(vocab("bluh"))
    vocab.freeze()
    print(vocab("bluh"))
    print(vocab("what?"))
    vocab.unfreeze()
    print(vocab("what?"))
    print(vocab("I'm", "afraid", "I", "can't", "do", "that", "dave", "!"))
    print(vocab(["all", "the", "work"]))
    print(vocab.get_sym(10))
    print(len(vocab))


