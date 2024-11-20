def custom(detection: 'Detection', tracked_object: 'TrackedObject') ->float:
    iou_score = iou(detection.points.reshape(1, 4), tracked_object.estimate
        .reshape(1, 4))[0][0]
    feature_score = feature_distance(detection, tracked_object)
    if iou_score > 0.98:
        return 1
    return 0.3 * iou_score + 0.7 * feature_score
