def training_loss(self, pred, target):
    assert pred.shape[-1
        ] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
        pred.shape[-1], self.length)
    assert pred.shape[0] == target.shape[0
        ], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.
        shape[0], target.shape[0])
    device = pred.device
    pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0
        ) * self.step
    pred_bin = pred[..., 1:1 + self.bin_count]
    diff_bin_target = torch.abs(target[..., None] - self.bins)
    _, bin_idx = torch.min(diff_bin_target, dim=-1)
    bin_bias = self.bins[bin_idx]
    bin_bias.requires_grad = False
    result = pred_reg + bin_bias
    target_bins = torch.full_like(pred_bin, self.cn, device=device)
    n = pred.shape[0]
    target_bins[range(n), bin_idx] = self.cp
    loss_bin = self.BCEbins(pred_bin, target_bins)
    if self.use_loss_regression:
        loss_regression = self.MSELoss(result, target)
        loss = loss_bin + loss_regression
    else:
        loss = loss_bin
    out_result = result.clamp(min=self.min, max=self.max)
    return loss, out_result
