def diou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor, reduction: str=
    'none', eps: float=1e-07) ->torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)
    assert (x2 >= x1).all(), 'bad box: x1 larger than x2'
    assert (y2 >= y1).all(), 'bad box: y1 larger than y2'
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)
    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = (xc2 - xc1) ** 2 + (yc2 - yc1) ** 2 + eps
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = (x_p - x_g) ** 2 + (y_p - y_g) ** 2
    loss = 1 - iou + distance / diag_len
    if reduction == 'mean':
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
