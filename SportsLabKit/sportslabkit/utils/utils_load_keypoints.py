def load_keypoints(keypoint_json):
    """Loads source and target keypoints from a JSON file.

    Args:
        keypoint_json (str): Path to JSON file containing keypoints.

    Returns:
        source_keypoints (np.ndarray): Source keypoints.
        target_keypoints (np.ndarray): Target keypoints.
    """
    with open(keypoint_json) as f:
        data = json.load(f)
    source_keypoints = []
    target_keypoints = []
    for key, value in data.items():
        source_kp = value
        target_kp = literal_eval(key)
        source_keypoints.append(source_kp)
        target_keypoints.append(target_kp)
    source_keypoints = np.array(source_keypoints)
    target_keypoints = np.array(target_keypoints)
    return source_keypoints, target_keypoints
