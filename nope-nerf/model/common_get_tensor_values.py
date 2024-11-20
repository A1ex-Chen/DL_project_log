def get_tensor_values(tensor, p, mode='nearest', scale=True, detach=True,
    detach_p=True, align_corners=False):
    """
    Returns values from tensor at given location p.

    Args:
        tensor (tensor): tensor of size B x C x H x W
        p (tensor): position values scaled between [-1, 1] and
            of size B x N x 2
        mode (str): interpolation mode
        scale (bool): whether to scale p from image coordinates to [-1, 1]
        detach (bool): whether to detach the output
        detach_p (bool): whether to detach p    
        align_corners (bool): whether to align corners for grid_sample
    """
    batch_size, _, h, w = tensor.shape
    if detach_p:
        p = p.detach()
    if scale:
        p[:, :, 0] = 2.0 * p[:, :, 0] / w - 1
        p[:, :, 1] = 2.0 * p[:, :, 1] / h - 1
    p = p.unsqueeze(1)
    values = torch.nn.functional.grid_sample(tensor, p, mode=mode,
        align_corners=align_corners)
    values = values.squeeze(2)
    if detach:
        values = values.detach()
    values = values.permute(0, 2, 1)
    return values
