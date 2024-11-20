def bbox_decode(self, anchor_points, pred_dist):
    if self.use_dfl:
        batch_size, n_anchors, _ = pred_dist.shape
        pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self
            .reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
    return dist2bbox(pred_dist, anchor_points)
