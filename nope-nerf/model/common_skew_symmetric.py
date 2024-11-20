def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O, -w2, w1], dim=-1), torch.stack([w2, O,
        -w0], dim=-1), torch.stack([-w1, w0, O], dim=-1)], dim=-2)
    return wx
