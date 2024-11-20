def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-05):
    """
    Parameters
    ----------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)
