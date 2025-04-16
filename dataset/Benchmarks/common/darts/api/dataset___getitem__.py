def __getitem__(self, idx):
    return self.dataset[self.indices[idx]]
