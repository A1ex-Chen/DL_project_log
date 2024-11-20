def create_keypoints_voting_distance(keypoint_distance_threshold: float,
    detection_threshold: float) ->Callable[['Detection', 'TrackedObject'],
    float]:
    """
    Construct a keypoint voting distance function configured with the thresholds.

    Count how many points in a detection match the with a tracked_object.
    A match is considered when distance between the points is < `keypoint_distance_threshold`
    and the score of the last_detection of the tracked_object is > `detection_threshold`.
    Notice the if multiple points are tracked, the ith point in detection can only match the ith
    point in the tracked object.

    Distance is 1 if no point matches and approximates 0 as more points are matched.

    Parameters
    ----------
    keypoint_distance_threshold: float
        Points closer than this threshold are considered a match.
    detection_threshold: float
        Detections and objects with score lower than this threshold are ignored.

    Returns
    -------
    Callable
        The distance funtion that must be passed to the Tracker.
    """

    def keypoints_voting_distance(detection: 'Detection', tracked_object:
        'TrackedObject') ->float:
        distances = np.linalg.norm(detection.points - tracked_object.
            estimate, axis=1)
        match_num = np.count_nonzero((distances <
            keypoint_distance_threshold) * (detection.scores >
            detection_threshold) * (tracked_object.last_detection.scores >
            detection_threshold))
        return 1 / (1 + match_num)
    return keypoints_voting_distance
