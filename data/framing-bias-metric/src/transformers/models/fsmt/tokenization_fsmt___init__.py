def __init__(self, langs=None, src_vocab_file=None, tgt_vocab_file=None,
    merges_file=None, do_lower_case=False, unk_token='<unk>', bos_token=
    '<s>', sep_token='</s>', pad_token='<pad>', **kwargs):
    super().__init__(langs=langs, src_vocab_file=src_vocab_file,
        tgt_vocab_file=tgt_vocab_file, merges_file=merges_file,
        do_lower_case=do_lower_case, unk_token=unk_token, bos_token=
        bos_token, sep_token=sep_token, pad_token=pad_token, **kwargs)
    self.src_vocab_file = src_vocab_file
    self.tgt_vocab_file = tgt_vocab_file
    self.merges_file = merges_file
    self.do_lower_case = do_lower_case
    self.cache_moses_punct_normalizer = dict()
    self.cache_moses_tokenizer = dict()
    self.cache_moses_detokenizer = dict()
    if langs and len(langs) == 2:
        self.src_lang, self.tgt_lang = langs
    else:
        raise ValueError(
            f"arg `langs` needs to be a list of 2 langs, e.g. ['en', 'ru'], but got {langs}. Usually that means that tokenizer can't find a mapping for the given model path in PRETRAINED_VOCAB_FILES_MAP, and other maps of this tokenizer."
            )
    with open(src_vocab_file, encoding='utf-8') as src_vocab_handle:
        self.encoder = json.load(src_vocab_handle)
    with open(tgt_vocab_file, encoding='utf-8') as tgt_vocab_handle:
        tgt_vocab = json.load(tgt_vocab_handle)
        self.decoder = {v: k for k, v in tgt_vocab.items()}
    with open(merges_file, encoding='utf-8') as merges_handle:
        merges = merges_handle.read().split('\n')[:-1]
    merges = [tuple(merge.split()[:2]) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
