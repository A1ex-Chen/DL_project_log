def _convert_keypoints(self, keypoints):
    if isinstance(keypoints, Keypoints):
        keypoints = keypoints.tensor
    keypoints = np.asarray(keypoints)
    return keypoints
