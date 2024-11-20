def calc_mean_std(feat: torch.Tensor, eps: float=1e-05) ->Tuple[torch.
    Tensor, torch.Tensor]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std
