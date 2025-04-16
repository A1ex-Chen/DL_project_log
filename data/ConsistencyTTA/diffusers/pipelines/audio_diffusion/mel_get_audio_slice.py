def get_audio_slice(self, slice: int=0) ->np.ndarray:
    """Get slice of audio.

        Args:
            slice (`int`): slice number of audio (out of get_number_of_slices())

        Returns:
            `np.ndarray`: audio as numpy array
        """
    return self.audio[self.slice_size * slice:self.slice_size * (slice + 1)]
