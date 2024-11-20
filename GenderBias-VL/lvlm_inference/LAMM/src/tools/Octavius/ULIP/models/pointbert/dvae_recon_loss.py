def recon_loss(self, ret, gt):
    whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret
    bs, g, _, _ = coarse.shape
    coarse = coarse.reshape(bs * g, -1, 3).contiguous()
    fine = fine.reshape(bs * g, -1, 3).contiguous()
    group_gt = group_gt.reshape(bs * g, -1, 3).contiguous()
    loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
    loss_fine_block = self.loss_func_cdl1(fine, group_gt)
    loss_recon = loss_coarse_block + loss_fine_block
    return loss_recon
