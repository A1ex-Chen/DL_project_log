def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    if clip:
        clip_boxes(x, (h - eps, w - eps))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2 / w
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2 / h
    y[..., 2] = (x[..., 2] - x[..., 0]) / w
    y[..., 3] = (x[..., 3] - x[..., 1]) / h
    return y
