def __call__(self, p, targets, imgs):
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device
        =device), torch.zeros(1, device=device)
    bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = (self.
        build_targets2(p[:self.nl], targets, imgs))
    bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.nl],
        targets, imgs)
    pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]
        ] for pp in p[:self.nl]]
    pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for
        pp in p[:self.nl]]
    for i in range(self.nl):
        pi = p[i]
        pi_aux = p[i + self.nl]
        b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
        b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i
            ], gis_aux[i]
        tobj = torch.zeros_like(pi[..., 0], device=device)
        tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)
        n = b.shape[0]
        if n:
            ps = pi[b, a, gj, gi]
            grid = torch.stack([gi, gj], dim=1)
            pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)
            selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
            selected_tbox[:, :2] -= grid
            iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)
            lbox += (1.0 - iou).mean()
            tobj[b, a, gj, gi] = 1.0 - self.gr + self.gr * iou.detach().clamp(0
                ).type(tobj.dtype)
            selected_tcls = targets[i][:, 1].long()
            if self.nc > 1:
                t = torch.full_like(ps[:, 5:], self.cn, device=device)
                t[range(n), selected_tcls] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t)
        n_aux = b_aux.shape[0]
        if n_aux:
            ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]
            grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
            pxy_aux = ps_aux[:, :2].sigmoid() * 2.0 - 0.5
            pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
            pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)
            selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
            selected_tbox_aux[:, :2] -= grid_aux
            iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=
                False, CIoU=True)
            lbox += 0.25 * (1.0 - iou_aux).mean()
            tobj_aux[b_aux, a_aux, gj_aux, gi_aux
                ] = 1.0 - self.gr + self.gr * iou_aux.detach().clamp(0).type(
                tobj_aux.dtype)
            selected_tcls_aux = targets_aux[i][:, 1].long()
            if self.nc > 1:
                t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)
                t_aux[range(n_aux), selected_tcls_aux] = self.cp
                lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)
        obji = self.BCEobj(pi[..., 4], tobj)
        obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
        lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i]
        if self.autobalance:
            self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach(
                ).item()
    if self.autobalance:
        self.balance = [(x / self.balance[self.ssi]) for x in self.balance]
    lbox *= self.hyp['box']
    lobj *= self.hyp['obj']
    lcls *= self.hyp['cls']
    bs = tobj.shape[0]
    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
