def get_batch(self, i, bptt=None):
    if bptt is None:
        bptt = self.bptt
    seq_len = min(bptt, self.data.size(0) - 1 - i)
    end_idx = i + seq_len
    beg_idx = max(0, i - self.ext_len)
    data = self.data[beg_idx:end_idx].to(self.device, non_blocking=True)
    target = self.data[i + 1:i + 1 + seq_len].to(self.device, non_blocking=True
        )
    if self.mem_len and self.warmup:
        warm = i >= self.warmup_elems
    else:
        warm = True
    return data, target, seq_len, warm
