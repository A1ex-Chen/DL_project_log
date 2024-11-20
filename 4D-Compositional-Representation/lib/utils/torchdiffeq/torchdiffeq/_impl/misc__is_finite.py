def _is_finite(tensor):
    _check = (tensor == float('inf')) + (tensor == float('-inf')
        ) + torch.isnan(tensor)
    return not _check.any()
