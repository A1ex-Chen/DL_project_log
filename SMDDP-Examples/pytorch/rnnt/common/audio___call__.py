def __call__(self, data, sample_rate):
    if self.discrete:
        rate = np.random.choice([self.min_rate, None, self.max_rate])
    else:
        rate = self._rng.uniform(self.min_rate, self.max_rate)
    if rate is not None:
        data._samples = sox.Transformer().speed(factor=rate).build_array(
            input_array=data._samples, sample_rate_in=sample_rate)
