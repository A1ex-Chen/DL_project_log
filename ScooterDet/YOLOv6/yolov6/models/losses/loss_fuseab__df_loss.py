def _df_loss(self, pred_dist, target):
    target_left = target.to(torch.long)
    target_right = target_left + 1
    weight_left = target_right.to(torch.float) - target
    weight_right = 1 - weight_left
    loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1),
        target_left.view(-1), reduction='none').view(target_left.shape
        ) * weight_left
    loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1),
        target_right.view(-1), reduction='none').view(target_left.shape
        ) * weight_right
    return (loss_left + loss_right).mean(-1, keepdim=True)
