def read_from_file(self, audio_file):
    audio = read_centered_wav(audio_file, target_sr=self.sr)
    audio = audio[:int(self.sr * self.target_length / 100)]
    audio = pad_short_audio(audio, min_samples=32000)
    return audio
