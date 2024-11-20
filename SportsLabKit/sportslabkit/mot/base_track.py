def track(self, sequence: (Iterable[Any] | np.ndarray)) ->Tracklet:
    if not isinstance(sequence, (Iterable, np.ndarray)):
        raise ValueError(
            "Input 'sequence' must be an iterable or numpy array of frames/batches"
            )
    self.reset()
    self.track_sequence(sequence)
    self.alive_tracklets = self.cleanup_tracklets(self.alive_tracklets)
    bbdf = self.to_bbdf()
    return bbdf
