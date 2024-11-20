@property
def seg_length(self):
    sr = self.sample_rate[-1] if isinstance(self.sample_rate, list
        ) else self.sample_rate
    return int(self.target_length * sr / 100), int(1000 * sr / 100)
