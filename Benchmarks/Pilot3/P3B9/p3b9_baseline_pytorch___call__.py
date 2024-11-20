def __call__(self, doc):
    return doc[:self.max_len]
