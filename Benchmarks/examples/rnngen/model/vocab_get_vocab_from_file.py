def get_vocab_from_file(fname):
    vocab = open(fname, 'r').readlines()
    vocab_ = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(),
        vocab)))
    vocab_c2i = {k: v for v, k in enumerate(vocab_)}
    vocab_i2c = {v: k for v, k in enumerate(vocab_)}

    def i2c(i):
        return vocab_i2c[i]

    def c2i(c):
        return vocab_c2i[c]
    return vocab, c2i, i2c, vocab_c2i, vocab_i2c
