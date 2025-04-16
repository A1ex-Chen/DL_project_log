def get_number_of_slices(self) ->int:
    """Get number of slices in audio.

        Returns:
            `int`: number of spectograms audio can be sliced into
        """
    return len(self.audio) // self.slice_size
