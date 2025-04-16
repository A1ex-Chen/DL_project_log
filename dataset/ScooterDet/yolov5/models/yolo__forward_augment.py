def _forward_augment(self, x):
    img_size = x.shape[-2:]
    s = [1, 0.83, 0.67]
    f = [None, 3, None]
    y = []
    for si, fi in zip(s, f):
        xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
        yi = self._forward_once(xi)[0]
        yi = self._descale_pred(yi, fi, si, img_size)
        y.append(yi)
    y = self._clip_augmented(y)
    return torch.cat(y, 1), None
