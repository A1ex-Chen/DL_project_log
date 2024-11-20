@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (
            query_boxes[k, 3] - query_boxes[k, 1] + eps)
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0],
                query_boxes[k, 0]) + eps
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1],
                    query_boxes[k, 1]) + eps
                if ih > 0:
                    ua = (boxes[n, 2] - boxes[n, 0] + eps) * (boxes[n, 3] -
                        boxes[n, 1] + eps) + box_area - iw * ih
                    overlaps[n, k] = iw * ih / ua
    return overlaps
