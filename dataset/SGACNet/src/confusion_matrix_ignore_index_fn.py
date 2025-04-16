def ignore_index_fn(iou_vector):
    if ignore_index >= len(iou_vector):
        raise ValueError(
            'ignore_index {} is larger than the length of IoU vector {}'.
            format(ignore_index, len(iou_vector)))
    indices = list(range(len(iou_vector)))
    indices.remove(ignore_index)
    return iou_vector[indices]
