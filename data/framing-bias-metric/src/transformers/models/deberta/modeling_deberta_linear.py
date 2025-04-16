def linear(w, b, x):
    if b is not None:
        return torch.matmul(x, w.t()) + b.t()
    else:
        return torch.matmul(x, w.t())
