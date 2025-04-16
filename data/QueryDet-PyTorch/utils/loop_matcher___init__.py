def __init__(self, thresholds: List[float], labels: List[int],
    allow_low_quality_matches: bool=False):
    thresholds = thresholds[:]
    assert thresholds[0] > 0
    thresholds.insert(0, -float('inf'))
    thresholds.append(float('inf'))
    assert all(low <= high for low, high in zip(thresholds[:-1], thresholds
        [1:]))
    assert all(l in [-1, 0, 1] for l in labels)
    assert len(labels) == len(thresholds) - 1
    self.low_quality_thrshold = 0.3
    self.thresholds = thresholds
    self.labels = labels
    self.allow_low_quality_matches = allow_low_quality_matches
