@staticmethod
def rescale(ori_shape, boxes, target_shape):
    """Rescale the output to the original image shape"""
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] -
        target_shape[0] * ratio) / 2
    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio
    boxes[:, 0].clamp_(0, target_shape[1])
    boxes[:, 1].clamp_(0, target_shape[0])
    boxes[:, 2].clamp_(0, target_shape[1])
    boxes[:, 3].clamp_(0, target_shape[0])
    return boxes
