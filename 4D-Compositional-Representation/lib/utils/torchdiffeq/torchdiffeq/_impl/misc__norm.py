def _norm(x):
    """Compute RMS norm."""
    if torch.is_tensor(x):
        return x.norm() / x.numel() ** 0.5
    else:
        return torch.sqrt(sum(x_.norm() ** 2 for x_ in x) / sum(x_.numel() for
            x_ in x))
