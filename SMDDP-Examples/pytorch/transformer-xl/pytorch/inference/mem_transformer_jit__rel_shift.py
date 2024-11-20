def _rel_shift(self, x, zero_triu: bool=False):
    zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.
        device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=3)
    x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
    x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)
    if zero_triu:
        ones = torch.ones((x.size(2), x.size(3)))
        x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
    return x
