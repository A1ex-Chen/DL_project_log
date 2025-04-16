def apply_segments(self, segments, M):
    """
        Apply affine to segments and generate new bboxes from segments.

        Args:
            segments (ndarray): list of segments, [num_samples, 500, 2].
            M (ndarray): affine matrix.

        Returns:
            new_segments (ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes after affine, [N, 4].
        """
    n, num = segments.shape[:2]
    if n == 0:
        return [], segments
    xy = np.ones((n * num, 3), dtype=segments.dtype)
    segments = segments.reshape(-1, 2)
    xy[:, :2] = segments
    xy = xy @ M.T
    xy = xy[:, :2] / xy[:, 2:3]
    segments = xy.reshape(n, -1, 2)
    bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in
        segments], 0)
    return bboxes, segments
