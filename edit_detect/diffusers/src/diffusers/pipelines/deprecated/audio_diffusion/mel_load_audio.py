def load_audio(self, audio_file: str=None, raw_audio: np.ndarray=None):
    """Load audio.

        Args:
            audio_file (`str`):
                An audio file that must be on disk due to [Librosa](https://librosa.org/) limitation.
            raw_audio (`np.ndarray`):
                The raw audio file as a NumPy array.
        """
    if audio_file is not None:
        self.audio, _ = librosa.load(audio_file, mono=True, sr=self.sr)
    else:
        self.audio = raw_audio
    if len(self.audio) < self.x_res * self.hop_length:
        self.audio = np.concatenate([self.audio, np.zeros((self.x_res *
            self.hop_length - len(self.audio),))])
