def __init__(self, thresholds: List[float], labels: List[int],
    allow_low_quality_matches: bool=False):
    """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
    thresholds = thresholds[:]
    assert thresholds[0] > 0
    thresholds.insert(0, -float('inf'))
    thresholds.append(float('inf'))
    assert all([(low <= high) for low, high in zip(thresholds[:-1],
        thresholds[1:])])
    assert all([(l in [-1, 0, 1]) for l in labels])
    assert len(labels) == len(thresholds) - 1
    self.thresholds = thresholds
    self.labels = labels
    self.allow_low_quality_matches = allow_low_quality_matches
