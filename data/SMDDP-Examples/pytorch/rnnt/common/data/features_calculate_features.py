def calculate_features(self, x, x_lens):
    pad_time = 0
    pad_freq = 0
    if self.pad_align_time > 0:
        if self.pad_to_max_duration:
            max_size = max(x.size(2), self.max_len)
            pad_amt = max_size % self.pad_align_time
            pad_time = self.pad_align_time - pad_amt if pad_amt > 0 else 0
            pad_time = max_size + pad_time - x.size(2)
        else:
            pad_amt = x.size(2) % self.pad_align_time
            pad_time = self.pad_align_time - pad_amt if pad_amt > 0 else 0
    if self.pad_align_freq > 0:
        pad_amt = x.size(1) % self.pad_align_freq
        pad_freq = self.pad_align_freq - pad_amt if pad_amt > 0 else 0
    x = nn.functional.pad(x, (0, pad_time, 0, pad_freq))
    return x, x_lens
