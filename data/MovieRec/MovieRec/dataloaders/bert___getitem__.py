def __getitem__(self, index):
    user = self.users[index]
    seq = self.u2seq[user]
    answer = self.u2answer[user]
    negs = self.negative_samples[user]
    candidates = answer + negs
    labels = [1] * len(answer) + [0] * len(negs)
    seq = seq + [self.mask_token]
    seq = seq[-self.max_len:]
    padding_len = self.max_len - len(seq)
    seq = [0] * padding_len + seq
    return torch.LongTensor(seq), torch.LongTensor(candidates
        ), torch.LongTensor(labels)
