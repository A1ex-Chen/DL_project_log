def __call__(self, p, targets):
    lcls = torch.zeros(1, device=self.device)
    lbox = torch.zeros(1, device=self.device)
    lobj = torch.zeros(1, device=self.device)
    tcls, tbox, indices, anchors = self.build_targets(p, targets)
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
        n = b.shape[0]
        if n:
            pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
            pxy = pxy.sigmoid() * 2 - 0.5
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)
            iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
            lbox += (1.0 - iou).mean()
            iou = iou.detach().clamp(0).type(tobj.dtype)
            if self.sort_obj_iou:
                j = iou.argsort()
                b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
            if self.gr < 1:
                iou = 1.0 - self.gr + self.gr * iou
            tobj[b, a, gj, gi] = iou
            if self.nc > 1:
                t = torch.full_like(pcls, self.cn, device=self.device)
                t[range(n), tcls[i]] = self.cp
                lcls += self.BCEcls(pcls, t)
        obji = self.BCEobj(pi[..., 4], tobj)
        lobj += obji * self.balance[i]
        if self.autobalance:
            self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach(
                ).item()
    if self.autobalance:
        self.balance = [(x / self.balance[self.ssi]) for x in self.balance]
    lbox *= self.hyp['box']
    lobj *= self.hyp['obj']
    lcls *= self.hyp['cls']
    bs = tobj.shape[0]
    return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
