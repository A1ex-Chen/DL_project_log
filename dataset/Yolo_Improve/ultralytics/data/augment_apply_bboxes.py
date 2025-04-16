def apply_bboxes(self, bboxes, M):
    """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
    n = len(bboxes)
    if n == 0:
        return bboxes
    xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
    xy = xy @ M.T
    xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n,
        8)
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=
        bboxes.dtype).reshape(4, n).T
