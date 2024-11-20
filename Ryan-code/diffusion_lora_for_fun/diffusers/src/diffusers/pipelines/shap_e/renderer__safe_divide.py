def _safe_divide(a, b, epsilon=1e-06):
    return a / torch.where(b < 0, b - epsilon, b + epsilon)
