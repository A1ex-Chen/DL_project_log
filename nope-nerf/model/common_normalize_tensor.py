def normalize_tensor(tensor, min_norm=1e-05, feat_dim=-1):
    """ Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    """
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True
        ), min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor
