def __init__(self, vocab_file, merges_file, unk_token='<unk>', **kwargs):
    super().__init__(unk_token=unk_token, **kwargs)
    with open(vocab_file, encoding='utf-8') as vocab_handle:
        self.encoder = json.load(vocab_handle)
    self.decoder = {v: k for k, v in self.encoder.items()}
    with open(merges_file, encoding='utf-8') as merges_handle:
        merges = merges_handle.read().split('\n')[1:-1]
    merges = [tuple(merge.split()) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
