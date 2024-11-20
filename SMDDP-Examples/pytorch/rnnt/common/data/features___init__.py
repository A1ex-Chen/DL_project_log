def __init__(self, optim_level, pad_align_time=8, pad_align_freq=0,
    pad_to_max_duration=False, max_duration=float('inf')):
    super(PadAlign, self).__init__(optim_level)
    self.pad_align_time = pad_align_time
    self.pad_align_freq = pad_align_freq
    self.pad_to_max_duration = pad_to_max_duration
    if pad_to_max_duration:
        self.max_len = 1 + math.ceil((max_duration * sample_rate - self.
            win_length) / self.hop_length)
