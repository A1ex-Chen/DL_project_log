def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1,
        largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([
            1, 1, self.topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk),
        is_in_topk)
    return is_in_topk.to(metrics.dtype)
