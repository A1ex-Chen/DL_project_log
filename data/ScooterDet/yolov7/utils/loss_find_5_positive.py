def find_5_positive(self, p, targets):
    na, nt = self.na, targets.shape[0]
    indices, anch = [], []
    gain = torch.ones(7, device=targets.device).long()
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(
        1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    g = 1.0
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=
        targets.device).float() * g
    for i in range(self.nl):
        anchors = self.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        t = targets * gain
        if nt:
            r = t[:, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1.0 / r).max(2)[0] < self.hyp['anchor_t']
            t = t[j]
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
            l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0
        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        gij = (gxy - offsets).long()
        gi, gj = gij.T
        a = t[:, 6].long()
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[
            2] - 1)))
        anch.append(anchors[a])
    return indices, anch
