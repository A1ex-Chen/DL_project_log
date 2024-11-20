@staticmethod
def _descale_pred(p, flips, scale, img_size, dim=1):
    """De-scale predictions following augmented inference (inverse operation)."""
    p[:, :4] /= scale
    x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    if flips == 2:
        y = img_size[0] - y
    elif flips == 3:
        x = img_size[1] - x
    return torch.cat((x, y, wh, cls), dim)
