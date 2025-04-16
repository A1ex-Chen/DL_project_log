def normalized__mean_euclidean_distance(detection: 'Detection',
    tracked_object: 'TrackedObject') ->float:
    """Normalized mean euclidean distance"""
    difference = (detection.points - tracked_object.estimate).astype(float)
    difference[:, 0] /= width
    difference[:, 1] /= height
    return np.linalg.norm(difference, axis=1).mean()
