def xyxy2cxcywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x
        )
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y
