def get_depth_consistency_loss(self, d1_proj, d2, d2_proj=None, d1=None):
    loss = self.l1_loss(d1_proj, d2) / float(d1_proj.shape[1])
    if d2_proj is not None:
        loss = 0.5 * loss + 0.5 * self.l1_loss(d2_proj, d1) / float(d2_proj
            .shape[1])
    return loss
