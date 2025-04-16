def create_normalized_mean_euclidean_distance(height: int, width: int
    ) ->Callable[['Detection', 'TrackedObject'], float]:
    """
    Construct a normalized mean euclidean distance function configured with the max height and width.

    The result distance is bound to [0, 1] where 1 indicates oposite corners of the image.

    Parameters
    ----------
    height: int
        Height of the image.
    width: int
        Width of the image.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """

    def normalized__mean_euclidean_distance(detection: 'Detection',
        tracked_object: 'TrackedObject') ->float:
        """Normalized mean euclidean distance"""
        difference = (detection.points - tracked_object.estimate).astype(float)
        difference[:, 0] /= width
        difference[:, 1] /= height
        return np.linalg.norm(difference, axis=1).mean()
    return normalized__mean_euclidean_distance
