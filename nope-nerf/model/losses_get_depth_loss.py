def get_depth_loss(self, depth_pred, depth_gt):
    if self.depth_loss_type == 'l1':
        loss = self.l1_loss(depth_pred, depth_gt) / float(depth_pred.shape[0])
    elif self.depth_loss_type == 'invariant':
        loss = self.depth_loss_dpt(depth_pred, depth_gt)
    return loss
