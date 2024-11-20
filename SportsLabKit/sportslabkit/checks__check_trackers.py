def _check_trackers(trackers: Sequence[Tracklet]) ->None:
    if not isinstance(trackers, Sequence):
        raise TypeError(
            f'trackers should be a sequence, but is {type(trackers).__name__}')
    if not all(isinstance(t, Tracklet) for t in trackers):
        raise TypeError(
            f'trackers should be a sequence of SingleObjectTracker, but contains {type(trackers[0]).__name__}.'
            )
