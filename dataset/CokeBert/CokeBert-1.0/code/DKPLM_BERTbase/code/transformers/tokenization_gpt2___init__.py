def __init__(self, vocab_file, merges_file, errors='replace', unk_token=
    '<|endoftext|>', bos_token='<|endoftext|>', eos_token='<|endoftext|>',
    **kwargs):
    super(GPT2Tokenizer, self).__init__(bos_token=bos_token, eos_token=
        eos_token, unk_token=unk_token, **kwargs)
    self.max_len_single_sentence = self.max_len
    self.max_len_sentences_pair = self.max_len
    self.encoder = json.load(open(vocab_file, encoding='utf-8'))
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.errors = errors
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
    bpe_merges = [tuple(merge.split()) for merge in bpe_data]
    self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    self.cache = {}
    self.pat = re.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )
