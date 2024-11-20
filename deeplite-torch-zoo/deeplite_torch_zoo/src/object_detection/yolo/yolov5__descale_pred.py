def _descale_pred(self, p, flips, scale, img_size):
    if self.inplace:
        p[..., :4] /= scale
        if flips == 2:
            p[..., 1] = img_size[0] - p[..., 1]
        elif flips == 3:
            p[..., 0] = img_size[1] - p[..., 0]
    else:
        x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4
            ] / scale
        if flips == 2:
            y = img_size[0] - y
        elif flips == 3:
            x = img_size[1] - x
        p = torch.cat((x, y, wh, p[..., 4:]), -1)
    return p
