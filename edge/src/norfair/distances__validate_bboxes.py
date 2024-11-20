def _validate_bboxes(bboxes: np.ndarray):
    """
    Validate that bounding boxes are well formed.
    """
    assert isinstance(bboxes, np.ndarray) and len(bboxes.shape
        ) == 2 and bboxes.shape[1
        ] == 4, f'Bounding boxes must be defined as np.array with (N, 4) shape, {bboxes} given'
    if not (all(bboxes[:, 0] < bboxes[:, 2]) and all(bboxes[:, 1] < bboxes[
        :, 3])):
        warning(
            'Incorrect bounding boxes. Check that x_min < x_max and y_min < y_max.'
            )
