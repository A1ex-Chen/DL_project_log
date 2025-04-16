def mean_manhattan(detection: 'Detection', tracked_object: 'TrackedObject'
    ) ->float:
    """
    Average manhattan distance between the points in detection and the estimates in tracked_object

    Given by:

    $$
    d(a, b) = \\frac{\\sum_{i=0}^N ||a_i - b_i||_1}{N}
    $$

    Where $||a||_1$ is the manhattan norm.

    Parameters
    ----------
    detection : Detection
        A detection.
    tracked_object : TrackedObject
        a tracked object.

    Returns
    -------
    float
        The distance.

    See Also
    --------
    [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
    """
    return np.linalg.norm(detection.points - tracked_object.estimate, ord=1,
        axis=1).mean()
