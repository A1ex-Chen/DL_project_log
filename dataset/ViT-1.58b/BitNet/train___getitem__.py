def __getitem__(self, index):
    rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
    full_seq = self.data[rand_start:rand_start + self.seq_len + 1].long()
    return full_seq.cuda()
