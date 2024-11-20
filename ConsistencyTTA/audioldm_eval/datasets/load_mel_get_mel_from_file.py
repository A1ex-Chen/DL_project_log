def get_mel_from_file(self, audio_file):
    audio = read_centered_wav(audio_file, self.sr)
    if self._stft is not None:
        melspec, energy = self.get_mel_from_wav(audio)
    else:
        melspec, energy = None, None
    return melspec, energy, audio
