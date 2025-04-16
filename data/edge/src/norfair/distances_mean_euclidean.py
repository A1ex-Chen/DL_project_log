def mean_euclidean(detection: 'Detection', tracked_object: 'TrackedObject'
    ) ->float:
    """
    Average euclidean distance between the points in detection and estimates in tracked_object.

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_2}{N}
    $$

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        A tracked object

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(detection.points - tracked_object.estimate, axis=1
        ).mean()
