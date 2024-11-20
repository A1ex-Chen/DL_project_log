def dynamic_range_decompression(x, C=1):
    """
    Parameters
    ----------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
