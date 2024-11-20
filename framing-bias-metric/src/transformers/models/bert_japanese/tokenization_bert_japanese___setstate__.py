def __setstate__(self, state):
    self.__dict__ = state
    if self.word_tokenizer_type == 'mecab':
        self.word_tokenizer = MecabTokenizer(do_lower_case=self.
            do_lower_case, never_split=self.never_split, **self.
            mecab_kwargs or {})
