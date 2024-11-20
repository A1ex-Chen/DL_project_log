def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
    max_len = self.bptt + max_deviation * std
    i = start
    while True:
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
        bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
        data, target, seq_len = self.get_batch(i, bptt)
        i += seq_len
        yield data, target, seq_len
        if i >= self.data.size(0) - 2:
            break
