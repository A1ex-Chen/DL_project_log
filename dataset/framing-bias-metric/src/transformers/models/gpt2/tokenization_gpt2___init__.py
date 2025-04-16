def __init__(self, vocab_file, merges_file, errors='replace', unk_token=
    '<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>',
    add_prefix_space=False, **kwargs):
    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False
        ) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False
        ) if isinstance(eos_token, str) else eos_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False
        ) if isinstance(unk_token, str) else unk_token
    super().__init__(errors=errors, unk_token=unk_token, bos_token=
        bos_token, eos_token=eos_token, add_prefix_space=add_prefix_space,
        **kwargs)
    with open(vocab_file, encoding='utf-8') as vocab_handle:
        self.encoder = json.load(vocab_handle)
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.errors = errors
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    with open(merges_file, encoding='utf-8') as merges_handle:
        bpe_merges = merges_handle.read().split('\n')[1:-1]
    bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
    self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    self.cache = {}
    self.add_prefix_space = add_prefix_space
    self.pat = re.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )
