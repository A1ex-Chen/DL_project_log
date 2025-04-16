def __call__(self, preds, targets, masks):
    p, proto = preds
    bs, nm, mask_h, mask_w = proto.shape
    lcls = torch.zeros(1, device=self.device)
    lbox = torch.zeros(1, device=self.device)
    lobj = torch.zeros(1, device=self.device)
    lseg = torch.zeros(1, device=self.device)
    tcls, tbox, indices, anchors, tidxs, xywhn = self.build_targets(p, targets)
    for i, pi in enumerate(p):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)
        n = b.shape[0]
        if n:
            pxy, pwh, _, pcls, pmask = pi[b, a, gj, gi].split((2, 2, 1,
                self.nc, nm), 1)
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
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode=
                    'nearest')[0]
            marea = xywhn[i][:, 2:].prod(1)
            mxyxy = xywh2xyxy(xywhn[i] * torch.tensor([mask_w, mask_h,
                mask_w, mask_h], device=self.device))
            for bi in b.unique():
                j = b == bi
                if self.overlap:
                    mask_gti = torch.where(masks[bi][None] == tidxs[i][j].
                        view(-1, 1, 1), 1.0, 0.0)
                else:
                    mask_gti = masks[tidxs[i]][j]
                lseg += self.single_mask_loss(mask_gti, pmask[j], proto[bi],
                    mxyxy[j], marea[j])
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
    lseg *= self.hyp['box'] / bs
    loss = lbox + lobj + lcls + lseg
    return loss * bs, torch.cat((lbox, lseg, lobj, lcls)).detach()
