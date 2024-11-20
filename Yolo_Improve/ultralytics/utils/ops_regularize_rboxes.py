def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    w_ = torch.where(w > h, w, h)
    h_ = torch.where(w > h, h, w)
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    return torch.stack([x, y, w_, h_, t], dim=-1)
