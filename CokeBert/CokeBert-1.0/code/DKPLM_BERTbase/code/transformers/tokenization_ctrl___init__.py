def __init__(self, vocab_file, merges_file, unk_token='<unk>', **kwargs):
    super(CTRLTokenizer, self).__init__(unk_token=unk_token, **kwargs)
    self.max_len_single_sentence = self.max_len
    self.max_len_sentences_pair = self.max_len
    self.encoder = json.load(open(vocab_file, encoding='utf-8'))
    self.decoder = {v: k for k, v in self.encoder.items()}
    merges = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
    merges = [tuple(merge.split()) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
