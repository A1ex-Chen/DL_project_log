@property
def keypoint_map(self) ->dict[tuple[int, int], tuple[int, int]]:
    """Get dictionary of pitch keypoints in pitch space to pixel space.

        Returns:
            Dict: dictionary of pitch keypoints in pitch space to pixel space.

        """
    if self.source_keypoints is None:
        return None
    return {tuple(key): value for key, value in zip(self.target_keypoints,
        self.source_keypoints)}
