def makepmask(self, mask, h, w, th, step):
    th = th - step * 0.005
    if 0.05 >= th:
        th = 0.05
    mask = torch.mean(mask, dim=0)
    mask = mask / mask.max().item()
    mask = torch.where(mask > th, 1, 0)
    mask = mask.float()
    mask = mask.view(1, *self.attnmaps_sizes[0])
    img = FF.to_pil_image(mask)
    img = img.resize((w, h))
    mask = FF.resize(mask, (h, w), interpolation=FF.InterpolationMode.
        NEAREST, antialias=None)
    lmask = mask
    mask = mask.reshape(h * w)
    mask = torch.where(mask > 0.1, 1, 0)
    return img, mask, lmask
