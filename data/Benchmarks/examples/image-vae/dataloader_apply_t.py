def apply_t(self, x):
    x = x + list(''.join([(char * (self.embedding_width - len(x))) for char in
        [' ']]))
    smi = self.one_hot_encoded_fn(x)
    return smi
