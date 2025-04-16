def __getitem__(self, index):
    return self.concat_dataset[self.index_mapping[index]]
