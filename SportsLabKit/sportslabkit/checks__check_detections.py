def _check_detections(detections: Sequence[Detection]) ->None:
    if not isinstance(detections, Sequence):
        raise TypeError(
            f'detections should be a sequence, but is {type(detections).__name__}.'
            )
    if not all(isinstance(d, Detection) for d in detections):
        raise TypeError(
            f'detections should be a sequence of Detection, but contains {type(detections[0]).__name__}.'
            )
