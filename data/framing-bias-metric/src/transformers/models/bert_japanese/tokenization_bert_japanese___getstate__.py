def __getstate__(self):
    state = dict(self.__dict__)
    if self.word_tokenizer_type == 'mecab':
        del state['word_tokenizer']
    return state
