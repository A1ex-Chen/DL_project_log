def audio_slice_to_image(self, slice: int) ->Image.Image:
    """Convert slice of audio to spectrogram.

        Args:
            slice (`int`):
                Slice number of audio to convert (out of `get_number_of_slices()`).

        Returns:
            `PIL Image`:
                A grayscale image of `x_res x y_res`.
        """
    S = librosa.feature.melspectrogram(y=self.get_audio_slice(slice), sr=
        self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.
        n_mels)
    log_S = librosa.power_to_db(S, ref=np.max, top_db=self.top_db)
    bytedata = (((log_S + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5
        ).astype(np.uint8)
    image = Image.fromarray(bytedata)
    return image
