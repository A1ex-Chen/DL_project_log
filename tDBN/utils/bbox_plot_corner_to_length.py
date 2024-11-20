def corner_to_length(bboxes):
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape([1, -1])
    ndim = bboxes.shape[1] // 2
    return np.concatenate([bboxes[:, :ndim], bboxes[:, ndim:] - bboxes[:, :
        ndim]], axis=1)
