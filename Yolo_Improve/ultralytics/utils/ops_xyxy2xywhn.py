def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1
        ] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x
        )
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2 / w
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2 / h
    y[..., 2] = (x[..., 2] - x[..., 0]) / w
    y[..., 3] = (x[..., 3] - x[..., 1]) / h
    return y
