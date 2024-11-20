def create_docs(self, length, num_vocab, num_docs):
    docs = [self.random_doc(length, num_vocab) for _ in range(num_docs)]
    return torch.stack(docs)
