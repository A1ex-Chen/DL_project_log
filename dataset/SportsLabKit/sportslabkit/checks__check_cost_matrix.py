def _check_cost_matrix(cost_matrix: np.ndarray, trackers, detections) ->None:
    if len(trackers) == 0 or len(detections) == 0:
        return
    if not isinstance(cost_matrix, np.ndarray):
        raise TypeError(
            f'cost_matrix should be a numpy array, but is {type(cost_matrix).__name__}.'
            )
    if len(cost_matrix.shape) != 2:
        raise ValueError(
            f'cost_matrix should be a 2D array, but is {len(cost_matrix.shape)}D.'
            )
    if cost_matrix.shape[0] != len(trackers):
        raise ValueError(
            f'cost_matrix should have {len(trackers)} rows, but has {cost_matrix.shape[0]}.'
            )
    if cost_matrix.shape[1] != len(detections):
        raise ValueError(
            f'cost_matrix should have {len(detections)} columns, but has {cost_matrix.shape[1]}.'
            )
