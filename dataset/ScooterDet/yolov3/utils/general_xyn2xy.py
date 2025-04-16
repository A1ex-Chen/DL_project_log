def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw
    y[..., 1] = h * x[..., 1] + padh
    return y
