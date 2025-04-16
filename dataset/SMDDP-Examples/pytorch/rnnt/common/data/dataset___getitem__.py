def __getitem__(self, index):
    s = self.samples[index]
    rn_indx = np.random.randint(len(s['audio_filepath']))
    duration = s['audio_duration'][rn_indx] if 'audio_duration' in s else 0
    offset = s.get('offset', 0)
    segment = AudioSegment(s['audio_filepath'][rn_indx], target_sr=self.
        sample_rate, offset=offset, duration=duration, trim=self.trim_silence)
    for p in self.perturbations:
        p.maybe_apply(segment, self.sample_rate)
    segment = torch.FloatTensor(segment.samples)
    return segment, torch.tensor(segment.shape[0]).int(), torch.tensor(s[
        'transcript']), torch.tensor(len(s['transcript'])).int()
