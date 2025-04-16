def __len__(self):
    return self.data.size(0) // self.seq_len
