def kpts_decode(self, bs, kpts):
    """Decodes keypoints."""
    ndim = self.kpt_shape[1]
    if self.export:
        y = kpts.view(bs, *self.kpt_shape, -1)
        a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
        if ndim == 3:
            a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
        return a.view(bs, self.nk, -1)
    else:
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3] = y[:, 2::3].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)
            ) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)
            ) * self.strides
        return y
