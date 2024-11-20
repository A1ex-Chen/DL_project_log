def compute_metric(self, trackers: Sequence[Tracklet], detections: Sequence
    [Detection]) ->np.ndarray:
    vectors1 = np.array([t.feature for t in trackers])
    vectors2 = np.array([d.feature for d in detections])
    return cdist(vectors1, vectors2, 'cosine') / 2
