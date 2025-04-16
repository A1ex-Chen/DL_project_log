def __getitem__(self, idx):
    data = self.load_data(self.files[idx])
    if not self.flatten:
        data = data.reshape(data.shape[0], -1, 2)
    if self.transform:
        data = self.transform(data)
    out_data = data[:self.split]
    out_label = data[self.split:]
    return out_data, out_label
