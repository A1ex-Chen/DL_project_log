def tokenize(self, data):
    """Tokenize a dataset"""
    for doc in tqdm(data):
        for token in doc:
            self.vocab.add_word(token)
    idss = []
    for doc in data:
        ids = []
        for token in doc:
            ids.append(self.vocab.word2idx[token])
        idss.append(torch.tensor(ids).type(torch.int64))
    return torch.stack(idss)
