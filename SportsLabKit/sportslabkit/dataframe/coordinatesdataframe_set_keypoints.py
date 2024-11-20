def set_keypoints(self, source_keypoints: (ArrayLike | None)=None,
    target_keypoints: (ArrayLike | None)=None, mapping: (Mapping | None)=
    None, mapping_file: (PathLike | None)=None) ->None:
    """Set the keypoints for the homography transformation. Make sure that
        the target keypoints are the pitch coordinates. Also each keypoint must
        be a tuple of (Lon, Lat) or (x, y) coordinates.

        Args:
            source_keypoints (Optional[ArrayLike], optional): Keypoints in pitch space. Defaults to None.
            target_keypoints (Optional[ArrayLike], optional): Keypoints in video space. Defaults to None.
        """
    if mapping_file is not None:
        with open(mapping_file) as f:
            mapping = json.load(f)
    if mapping is not None:
        target_keypoints, source_keypoints = [], []
        for target_kp, source_kp in mapping.items():
            if isinstance(target_kp, str):
                target_kp = literal_eval(target_kp)
            if isinstance(source_kp, str):
                source_kp = literal_eval(source_kp)
            target_keypoints.append(target_kp)
            source_keypoints.append(source_kp)
    self.source_keypoints = np.array(source_keypoints)
    self.target_keypoints = np.array(target_keypoints)
