def __init__(self, vocab_file, merges_file, unk_token='<unk>', **kwargs):
    super(OpenAIGPTTokenizer, self).__init__(unk_token=unk_token, **kwargs)
    self.max_len_single_sentence = self.max_len
    self.max_len_sentences_pair = self.max_len
    try:
        import ftfy
        from spacy.lang.en import English
        _nlp = English()
        self.nlp = _nlp.Defaults.create_tokenizer(_nlp)
        self.fix_text = ftfy.fix_text
    except ImportError:
        logger.warning(
            'ftfy or spacy is not installed using BERT BasicTokenizer instead of SpaCy & ftfy.'
            )
        self.nlp = BasicTokenizer(do_lower_case=True)
        self.fix_text = None
    self.encoder = json.load(open(vocab_file, encoding='utf-8'))
    self.decoder = {v: k for k, v in self.encoder.items()}
    merges = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
    merges = [tuple(merge.split()) for merge in merges]
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {}
