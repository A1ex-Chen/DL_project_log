def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
    """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
    topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1,
        largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps
            ).expand_as(topk_idxs)
    topk_idxs.masked_fill_(~topk_mask, 0)
    count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=
        topk_idxs.device)
    ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=
        topk_idxs.device)
    for k in range(self.topk):
        count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
    count_tensor.masked_fill_(count_tensor > 1, 0)
    return count_tensor.to(metrics.dtype)
