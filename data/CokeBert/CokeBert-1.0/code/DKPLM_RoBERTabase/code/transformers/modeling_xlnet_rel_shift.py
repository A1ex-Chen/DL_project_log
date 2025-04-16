@staticmethod
def rel_shift(x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = x.shape
    x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
    x = x[1:, ...]
    x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
    x = torch.index_select(x, 1, torch.arange(klen, device=x.device, dtype=
        torch.long))
    return x
