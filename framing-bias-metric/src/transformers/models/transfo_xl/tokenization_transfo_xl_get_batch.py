def get_batch(self, i, bptt=None):
    if bptt is None:
        bptt = self.bptt
    seq_len = min(bptt, self.data.size(0) - 1 - i)
    end_idx = i + seq_len
    beg_idx = max(0, i - self.ext_len)
    data = self.data[beg_idx:end_idx]
    target = self.data[i + 1:i + 1 + seq_len]
    data_out = data.transpose(0, 1).contiguous().to(self.device)
    target_out = target.transpose(0, 1).contiguous().to(self.device)
    return data_out, target_out, seq_len
