def iou_pytorch(cm, ignore_index=None):
    if not isinstance(cm, ConfusionMatrixPytorch):
        raise TypeError(
            'Argument cm should be instance of ConfusionMatrix, but given {}'
            .format(type(cm)))
    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <=
            ignore_index < cm.num_classes):
            raise ValueError(
                'ignore_index should be non-negative integer, but given {}'
                .format(ignore_index))
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError(
                    'ignore_index {} is larger than the length of IoU vector {}'
                    .format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]
        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou
