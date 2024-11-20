def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
    """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
    d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] -
        gt_kpts[..., 1]).pow(2)
    kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) +
        1e-09)
    e = d / ((2 * self.sigmas).pow(2) * (area + 1e-09) * 2)
    return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)
        ).mean()
