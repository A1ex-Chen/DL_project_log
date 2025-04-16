def _rel_shift(self, x):
    zero_pad_shape = (x.size(0), 1) + x.size()[2:]
    zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
    x_padded = x_padded.view(*x_padded_shape)
    x = x_padded[1:].view_as(x)
    return x
