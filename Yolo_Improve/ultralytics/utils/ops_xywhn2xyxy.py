def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1
        ] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x
        )
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh
    return y
