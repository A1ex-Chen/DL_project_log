def build_targets(self, p, targets):
    na, nt = self.na, targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=self.device)
    ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)
    g = 0.5
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=
        self.device).float() * g
    for i in range(self.nl):
        anchors, shape = self.anchors[i], p[i].shape
        gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
        t = targets * gain
        if nt:
            r = t[..., 4:6] / anchors[:, None]
            j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
            t = t[j]
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxi % 1 < g) & (gxi > 1)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0
        bc, gxy, gwh, a = t.chunk(4, 1)
        a, (b, c) = a.long().view(-1), bc.long().T
        gij = (gxy - offsets).long()
        gi, gj = gij.T
        indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, 
            shape[3] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)
    return tcls, tbox, indices, anch
