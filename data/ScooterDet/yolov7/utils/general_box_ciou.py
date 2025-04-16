def box_ciou(box1, box2, eps: float=1e-07):
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:,
        None, :2], box2[:, :2])).clamp(0).prod(2)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    lti = torch.min(box1[:, None, :2], box2[:, :2])
    rbi = torch.max(box1[:, None, 2:], box2[:, 2:])
    whi = (rbi - lti).clamp(min=0)
    diagonal_distance_squared = whi[:, :, 0] ** 2 + whi[:, :, 1] ** 2 + eps
    x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
    y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
    x_g = (box2[:, 0] + box2[:, 2]) / 2
    y_g = (box2[:, 1] + box2[:, 3]) / 2
    centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2
    w_pred = box1[:, None, 2] - box1[:, None, 0]
    h_pred = box1[:, None, 3] - box1[:, None, 1]
    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]
    v = 4 / torch.pi ** 2 * torch.pow(torch.atan(w_gt / h_gt) - torch.atan(
        w_pred / h_pred), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return (iou - centers_distance_squared / diagonal_distance_squared - 
        alpha * v)
