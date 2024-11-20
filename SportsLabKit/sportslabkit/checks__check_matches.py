def _check_matches(matches: np.ndarray, trackers: Sequence[Tracklet],
    detections: Sequence[Detection]) ->None:
    if not isinstance(matches, np.ndarray):
        raise TypeError(
            f'matches should be a numpy array, but is {type(matches).__name__}.'
            )
    if len(matches.shape) != 2:
        raise ValueError(
            f'matches should be a 2D array, but is {len(matches.shape)}D.')
    if matches.shape[1] != 2:
        raise ValueError(
            f'matches should have 2 columns, but has {matches.shape[1]}.')
    if np.any(matches[:, 0] >= len(trackers)):
        raise ValueError(
            'matches contains rows with tracker index greater than number of trackers.'
            )
    if np.any(matches[:, 1] >= len(detections)):
        raise ValueError(
            'matches contains rows with detection index greater than number of detections.'
            )
