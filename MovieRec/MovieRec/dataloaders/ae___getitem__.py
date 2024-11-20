def __getitem__(self, index):
    return self.input_data[index], self.label_data[index]
