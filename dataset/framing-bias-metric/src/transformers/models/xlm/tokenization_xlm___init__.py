def __init__(self, vocab_file, merges_file, unk_token='<unk>', bos_token=
    '<s>', sep_token='</s>', pad_token='<pad>', cls_token='</s>',
    mask_token='<special1>', additional_special_tokens=['<special0>',
    '<special1>', '<special2>', '<special3>', '<special4>', '<special5>',
    '<special6>', '<special7>', '<special8>', '<special9>'], lang2id=None,
    id2lang=None, do_lowercase_and_remove_accent=True, **kwargs):
    super().__init__(unk_token=unk_token, bos_token=bos_token, sep_token=
        sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=
        mask_token, additional_special_tokens=additional_special_tokens,
        lang2id=lang2id, id2lang=id2lang, do_lowercase_and_remove_accent=
        do_lowercase_and_remove_accent, **kwargs)
    self.cache_moses_punct_normalizer = dict()
    self.cache_moses_tokenizer = dict()
    self.lang_with_custom_tokenizer = set(['zh', 'th', 'ja'])
    self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
    self.lang2id = lang2id
    self.id2lang = id2lang
    if lang2id is not None and id2lang is not None:
        assert len(lang2id) == len(id2lang)
    self.ja_word_tokenizer = None
    self.zh_word_tokenizer = None
    with open(vocab_file, encoding='utf-8') as vocab_handle:
        self.encoder = json.load(vocab_handle)
    self.decoder = {v: k for k, v in self.encoder.items()}
    with open(merges_file, encoding='utf-8') as merges_handle:
        merges = merges_handle.read().split('\n')[:-1]
    merges = [tuple(merge.split()[:2]) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
