@staticmethod
def dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask_inds):
    device = cost.device
    matching_matrix = torch.zeros(cost.shape, dtype=torch.uint8, device=device)
    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    dynamic_ks = topk_ious.sum(1).int().clamp_min_(1)
    if num_gt > 3:
        min_k, max_k = torch._aminmax(dynamic_ks)
        min_k, max_k = min_k.item(), max_k.item()
        if min_k != max_k:
            offsets = torch.arange(0, matching_matrix.shape[0] *
                matching_matrix.shape[1], step=matching_matrix.shape[1],
                dtype=torch.int, device=device)[:, None]
            masks = torch.arange(0, max_k, dtype=dynamic_ks.dtype, device=
                device)[None, :].expand(num_gt, max_k) < dynamic_ks[:, None]
            _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
            pos_idxes.add_(offsets)
            pos_idxes = torch.masked_select(pos_idxes, masks)
            matching_matrix.view(-1).index_fill_(0, pos_idxes, 1)
            del topk_ious, dynamic_ks, pos_idxes, offsets, masks
        else:
            _, pos_idxes = torch.topk(cost, k=max_k, dim=1, largest=False)
            matching_matrix.scatter_(1, pos_idxes, 1)
            del topk_ious, dynamic_ks
    else:
        ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx
    anchor_matching_gt = matching_matrix.sum(0)
    anchor_matching_one_more_gt_mask = anchor_matching_gt > 1
    anchor_matching_one_more_gt_inds = torch.nonzero(
        anchor_matching_one_more_gt_mask)
    if anchor_matching_one_more_gt_inds.shape[0] > 0:
        anchor_matching_one_more_gt_inds = anchor_matching_one_more_gt_inds[
            ..., 0]
        _, cost_argmin = torch.min(cost.index_select(1,
            anchor_matching_one_more_gt_inds), dim=0)
        matching_matrix.index_fill_(1, anchor_matching_one_more_gt_inds, 0)
        matching_matrix[cost_argmin, anchor_matching_one_more_gt_inds] = 1
        fg_mask_inboxes = matching_matrix.any(dim=0)
        fg_mask_inboxes_inds = torch.nonzero(fg_mask_inboxes)[..., 0]
    else:
        fg_mask_inboxes_inds = torch.nonzero(anchor_matching_gt)[..., 0]
    num_fg = fg_mask_inboxes_inds.shape[0]
    matched_gt_inds = matching_matrix.index_select(1, fg_mask_inboxes_inds
        ).argmax(0)
    fg_mask_inds = fg_mask_inds[fg_mask_inboxes_inds]
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious_this_matching = pair_wise_ious.index_select(1,
        fg_mask_inboxes_inds).gather(dim=0, index=matched_gt_inds[None, :])
    return (num_fg, gt_matched_classes, pred_ious_this_matching,
        matched_gt_inds, fg_mask_inds)
