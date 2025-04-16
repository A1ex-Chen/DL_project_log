def iou(candidates: np.ndarray, objects: np.ndarray) ->np.ndarray:
    """
    Calculate IoU between two sets of bounding boxes. Both sets of boxes are expected
    to be in `[x_min, y_min, x_max, y_max]` format.

    Normal IoU is 1 when the boxes are the same and 0 when they don't overlap,
    to transform that into a distance that makes sense we return `1 - iou`.

    Parameters
    ----------
    candidates : numpy.ndarray
        (N, 4) numpy.ndarray containing candidates bounding boxes.
    objects : numpy.ndarray
        (K, 4) numpy.ndarray containing objects bounding boxes.

    Returns
    -------
    numpy.ndarray
        (N, K) numpy.ndarray of `1 - iou` between candidates and objects.
    """
    _validate_bboxes(candidates)
    area_candidates = _boxes_area(candidates.T)
    area_objects = _boxes_area(objects.T)
    top_left = np.maximum(candidates[:, None, :2], objects[:, :2])
    bottom_right = np.minimum(candidates[:, None, 2:], objects[:, 2:])
    area_intersection = np.prod(np.clip(bottom_right - top_left, a_min=0,
        a_max=None), 2)
    return 1 - area_intersection / (area_candidates[:, None] + area_objects -
        area_intersection)
