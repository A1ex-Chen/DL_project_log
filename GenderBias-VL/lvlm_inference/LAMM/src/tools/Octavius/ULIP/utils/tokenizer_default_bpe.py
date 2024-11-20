@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
        'bpe_simple_vocab_16e6.txt.gz')
