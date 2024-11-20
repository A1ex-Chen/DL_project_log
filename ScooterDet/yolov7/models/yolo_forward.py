def forward(self, x, augment=False, profile=False):
    if augment:
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.
                max()))
            yi = self.forward_once(xi)[0]
            yi[..., :4] /= si
            if fi == 2:
                yi[..., 1] = img_size[0] - yi[..., 1]
            elif fi == 3:
                yi[..., 0] = img_size[1] - yi[..., 0]
            y.append(yi)
        return torch.cat(y, 1), None
    else:
        return self.forward_once(x, profile)
