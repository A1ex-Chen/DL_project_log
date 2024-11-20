def build_targets(self, p, targets):
    na, nt = self.na, targets.shape[0]
    tcls, tbox, indices, anch, tidxs, xywhn = [], [], [], [], [], []
    gain = torch.ones(8, device=self.device)
    ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
    if self.overlap:
        batch = p[0].shape[0]
        ti = []
        for i in range(batch):
            num = (targets[:, 0] == i).sum()
            ti.append(torch.arange(num, device=self.device).float().view(1,
                num).repeat(na, 1) + 1)
        ti = torch.cat(ti, 1)
    else:
        ti = torch.arange(nt, device=self.device).float().view(1, nt).repeat(na
            , 1)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None], ti[...,
        None]), 2)
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
        bc, gxy, gwh, at = t.chunk(4, 1)
        (a, tidx), (b, c) = at.long().T, bc.long().T
        gij = (gxy - offsets).long()
        gi, gj = gij.T
        indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, 
            shape[3] - 1)))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)
        tidxs.append(tidx)
        xywhn.append(torch.cat((gxy, gwh), 1) / gain[2:6])
    return tcls, tbox, indices, anch, tidxs, xywhn
