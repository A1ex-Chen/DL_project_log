def __getstate__(self):
    state = self.__dict__.copy()
    del state['tokenizer']
    return state
