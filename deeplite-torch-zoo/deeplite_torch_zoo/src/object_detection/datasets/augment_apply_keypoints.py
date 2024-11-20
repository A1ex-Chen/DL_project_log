def apply_keypoints(self, keypoints, M):
    """
        Apply affine to keypoints.

        Args:
            keypoints (ndarray): keypoints, [N, 17, 3].
            M (ndarray): affine matrix.

        Return:
            new_keypoints (ndarray): keypoints after affine, [N, 17, 3].
        """
    n, nkpt = keypoints.shape[:2]
    if n == 0:
        return keypoints
    xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
    visible = keypoints[..., 2].reshape(n * nkpt, 1)
    xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
    xy = xy @ M.T
    xy = xy[:, :2] / xy[:, 2:3]
    out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (
        xy[:, 1] > self.size[1])
    visible[out_mask] = 0
    return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)
