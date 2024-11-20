def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh
    return y
