def cross_entropy(input, target, *, reduction='mean', **kwargs):
    """
    Same as `torch.nn.functional.cross_entropy`, but returns 0 (instead of nan)
    for empty inputs.
    """
    if target.numel() == 0 and reduction == 'mean':
        return input.sum() * 0.0
    return F.cross_entropy(input, target, reduction=reduction, **kwargs)
