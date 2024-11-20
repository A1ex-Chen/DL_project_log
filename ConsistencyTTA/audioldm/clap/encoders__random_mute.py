def _random_mute(self, waveform):
    t_steps = waveform.size(-1)
    for i in range(waveform.size(0)):
        mute_size = int(self.random_uniform(0, end=int(t_steps * self.
            max_random_mute_portion)))
        mute_start = int(self.random_uniform(0, t_steps - mute_size))
        waveform[i, mute_start:mute_start + mute_size] = 0
    return waveform
