def __getitem__(self, idx):
    if isinstance(idx, slice):
        return SingleSentenceClassificationProcessor(labels=self.labels,
            examples=self.examples[idx])
    return self.examples[idx]
