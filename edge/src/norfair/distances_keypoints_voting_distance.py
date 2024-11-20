def keypoints_voting_distance(detection: 'Detection', tracked_object:
    'TrackedObject') ->float:
    distances = np.linalg.norm(detection.points - tracked_object.estimate,
        axis=1)
    match_num = np.count_nonzero((distances < keypoint_distance_threshold) *
        (detection.scores > detection_threshold) * (tracked_object.
        last_detection.scores > detection_threshold))
    return 1 / (1 + match_num)
