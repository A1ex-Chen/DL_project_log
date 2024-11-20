def track(self, sequence: (Iterable[Any] | np.ndarray)) ->Tracklet:
    if not isinstance(sequence, (Iterable, np.ndarray)):
        raise ValueError(
            "Input 'sequence' must be an iterable or numpy array of frames/batches"
            )
    self.pre_track()
    for i in range(0, len(sequence) - self.window_size + 1, self.step_size):
        logger.debug(f'Processing frames {i} to {i + self.window_size}')
        self.process_sequence_item(sequence[i:i + self.window_size])
    self.post_track()
    return self.tracklet
