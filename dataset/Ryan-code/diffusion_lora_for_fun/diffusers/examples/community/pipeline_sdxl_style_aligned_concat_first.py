def concat_first(feat: torch.Tensor, dim: int=2, scale: float=1.0
    ) ->torch.Tensor:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)
