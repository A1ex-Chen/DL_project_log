def __init__(self, vocab_file, merges_file, bos_token='__start__',
    eos_token='__end__', unk_token='__unk__', pad_token='__null__', **kwargs):
    super().__init__(unk_token=unk_token, bos_token=bos_token, eos_token=
        eos_token, pad_token=pad_token, **kwargs)
    with open(vocab_file, encoding='utf-8') as vocab_handle:
        self.encoder = json.load(vocab_handle)
    self.decoder = {v: k for k, v in self.encoder.items()}
    with open(merges_file, encoding='utf-8') as merges_handle:
        merges = merges_handle.read().split('\n')[1:-1]
    merges = [tuple(merge.split()) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
