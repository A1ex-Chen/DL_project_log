def __getitem__(self, idx):
    shard = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
    x_list, y = self.get_slice(self.batch_size, single=self.single,
        partial_index=shard)
    return x_list, y
