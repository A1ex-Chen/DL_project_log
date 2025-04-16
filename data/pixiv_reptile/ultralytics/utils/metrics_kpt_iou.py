def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-07):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] -
        kpt2[..., 1]).pow(2)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)
    kpt_mask = kpt1[..., 2] != 0
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:,
        None] + eps)
