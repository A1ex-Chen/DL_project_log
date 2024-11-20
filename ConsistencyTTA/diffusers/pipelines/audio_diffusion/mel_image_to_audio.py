def image_to_audio(self, image: Image.Image) ->np.ndarray:
    """Converts spectrogram to audio.

        Args:
            image (`PIL Image`): x_res x y_res grayscale image

        Returns:
            audio (`np.ndarray`): raw audio
        """
    bytedata = np.frombuffer(image.tobytes(), dtype='uint8').reshape((image
        .height, image.width))
    log_S = bytedata.astype('float') * self.top_db / 255 - self.top_db
    S = librosa.db_to_power(log_S)
    audio = librosa.feature.inverse.mel_to_audio(S, sr=self.sr, n_fft=self.
        n_fft, hop_length=self.hop_length, n_iter=self.n_iter)
    return audio
