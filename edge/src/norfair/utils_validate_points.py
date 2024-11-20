def validate_points(points: np.ndarray) ->np.array:
    if len(points.shape) == 1:
        points = points[np.newaxis, ...]
    elif len(points.shape) > 2:
        print_detection_error_message_and_exit(points)
    return points
