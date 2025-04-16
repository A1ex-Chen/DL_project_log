def __getitem__(self, index):
    return self.concat_input[index], self.label[index]
