def set_resolution(self, x_res: int, y_res: int):
    """Set resolution.

        Args:
            x_res (`int`): x resolution of spectrogram (time)
            y_res (`int`): y resolution of spectrogram (frequency bins)
        """
    self.x_res = x_res
    self.y_res = y_res
    self.n_mels = self.y_res
    self.slice_size = self.x_res * self.hop_length - 1
