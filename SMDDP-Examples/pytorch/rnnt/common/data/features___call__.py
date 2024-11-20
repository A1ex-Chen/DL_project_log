def __call__(self, x):
    audio, audio_lens = x
    if self.optim_level == 1:
        with amp.disable_casts():
            return self.calculate_features(audio, audio_lens)
    else:
        return self.calculate_features(audio, audio_lens)
