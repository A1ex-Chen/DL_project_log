def define_graph(self):
    audio, label = self.read()
    if not self.train or self.speed_perturbation_coeffs is None:
        audio, sr = self.decode(audio)
    else:
        resample_coeffs = self.speed_perturbation_coeffs() * self.sample_rate
        audio, sr = self.decode(audio, sample_rate=resample_coeffs)
    if self.do_remove_silence:
        audio = self._remove_silence(audio)
    if self.synthetic_seq_len is not None:
        audio = self.constant()
    if self.preprocessing_device == 'gpu':
        audio = audio.gpu()
    if self.dither_coeff != 0.0:
        audio = audio + self.normal_distribution(audio) * self.dither_coeff
    audio = self.preemph(audio)
    audio = self.spectrogram(audio)
    audio = self.mel_fbank(audio)
    audio = self.log_features(audio)
    audio_len = self.get_shape(audio)
    audio = self.normalize(audio)
    audio = self.pad(audio)
    return audio.gpu(), label, audio_len.gpu()
