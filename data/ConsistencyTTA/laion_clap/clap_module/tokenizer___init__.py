def __init__(self, bpe_path: str=default_bpe(), special_tokens=None):
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    merges = gzip.open(bpe_path).read().decode('utf-8').split('\n')
    merges = merges[1:49152 - 256 - 2 + 1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [(v + '</w>') for v in vocab]
    for merge in merges:
        vocab.append(''.join(merge))
    if not special_tokens:
        special_tokens = ['<start_of_text>', '<end_of_text>']
    else:
        special_tokens = ['<start_of_text>', '<end_of_text>'] + special_tokens
    vocab.extend(special_tokens)
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {t: t for t in special_tokens}
    special = '|'.join(special_tokens)
    self.pat = re.compile(special +
        "|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
        re.IGNORECASE)
    self.vocab_size = len(self.encoder)
    self.all_special_ids = [self.encoder[t] for t in special_tokens]
