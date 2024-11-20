def build_targets2(self, p, targets, imgs):
    indices, anch = self.find_5_positive(p, targets)
    matching_bs = [[] for pp in p]
    matching_as = [[] for pp in p]
    matching_gjs = [[] for pp in p]
    matching_gis = [[] for pp in p]
    matching_targets = [[] for pp in p]
    matching_anchs = [[] for pp in p]
    nl = len(p)
    for batch_idx in range(p[0].shape[0]):
        b_idx = targets[:, 0] == batch_idx
        this_target = targets[b_idx]
        if this_target.shape[0] == 0:
            continue
        txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
        txyxy = xywh2xyxy(txywh)
        pxyxys = []
        p_cls = []
        p_obj = []
        from_which_layer = []
        all_b = []
        all_a = []
        all_gj = []
        all_gi = []
        all_anch = []
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            idx = b == batch_idx
            b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
            all_b.append(b)
            all_a.append(a)
            all_gj.append(gj)
            all_gi.append(gi)
            all_anch.append(anch[i][idx])
            from_which_layer.append(torch.ones(size=(len(b),)) * i)
            fg_pred = pi[b, a, gj, gi]
            p_obj.append(fg_pred[:, 4:5])
            p_cls.append(fg_pred[:, 5:])
            grid = torch.stack([gi, gj], dim=1)
            pxy = (fg_pred[:, :2].sigmoid() * 2.0 - 0.5 + grid) * self.stride[i
                ]
            pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx
                ] * self.stride[i]
            pxywh = torch.cat([pxy, pwh], dim=-1)
            pxyxy = xywh2xyxy(pxywh)
            pxyxys.append(pxyxy)
        pxyxys = torch.cat(pxyxys, dim=0)
        if pxyxys.shape[0] == 0:
            continue
        p_obj = torch.cat(p_obj, dim=0)
        p_cls = torch.cat(p_cls, dim=0)
        from_which_layer = torch.cat(from_which_layer, dim=0)
        all_b = torch.cat(all_b, dim=0)
        all_a = torch.cat(all_a, dim=0)
        all_gj = torch.cat(all_gj, dim=0)
        all_gi = torch.cat(all_gi, dim=0)
        all_anch = torch.cat(all_anch, dim=0)
        pair_wise_iou = box_iou(txyxy, pxyxys)
        pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-08)
        top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]
            ), dim=1)
        dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)
        gt_cls_per_image = F.one_hot(this_target[:, 1].to(torch.int64), self.nc
            ).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)
        num_gt = this_target.shape[0]
        cls_preds_ = p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_(
            ) * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        y = cls_preds_.sqrt_()
        pair_wise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(y /
            (1 - y)), gt_cls_per_image, reduction='none').sum(-1)
        del cls_preds_
        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        matching_matrix = torch.zeros_like(cost, device='cpu')
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item
                (), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del top_k, dynamic_ks
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        from_which_layer = from_which_layer[fg_mask_inboxes]
        all_b = all_b[fg_mask_inboxes]
        all_a = all_a[fg_mask_inboxes]
        all_gj = all_gj[fg_mask_inboxes]
        all_gi = all_gi[fg_mask_inboxes]
        all_anch = all_anch[fg_mask_inboxes]
        this_target = this_target[matched_gt_inds]
        for i in range(nl):
            layer_idx = from_which_layer == i
            matching_bs[i].append(all_b[layer_idx])
            matching_as[i].append(all_a[layer_idx])
            matching_gjs[i].append(all_gj[layer_idx])
            matching_gis[i].append(all_gi[layer_idx])
            matching_targets[i].append(this_target[layer_idx])
            matching_anchs[i].append(all_anch[layer_idx])
    for i in range(nl):
        if matching_targets[i] != []:
            matching_bs[i] = torch.cat(matching_bs[i], dim=0)
            matching_as[i] = torch.cat(matching_as[i], dim=0)
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
            matching_gis[i] = torch.cat(matching_gis[i], dim=0)
            matching_targets[i] = torch.cat(matching_targets[i], dim=0)
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
        else:
            matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.
                int64)
            matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.
                int64)
            matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch
                .int64)
            matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch
                .int64)
            matching_targets[i] = torch.tensor([], device='cuda:0', dtype=
                torch.int64)
            matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=
                torch.int64)
    return (matching_bs, matching_as, matching_gjs, matching_gis,
        matching_targets, matching_anchs)
