def select_topk_candidates(self, distances, n_level_bboxes, mask_gt):
    mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
    level_distances = torch.split(distances, n_level_bboxes, dim=-1)
    is_in_candidate_list = []
    candidate_idxs = []
    start_idx = 0
    for per_level_distances, per_level_boxes in zip(level_distances,
        n_level_bboxes):
        end_idx = start_idx + per_level_boxes
        selected_k = min(self.topk, per_level_boxes)
        _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-
            1, largest=False)
        candidate_idxs.append(per_level_topk_idxs + start_idx)
        per_level_topk_idxs = torch.where(mask_gt, per_level_topk_idxs,
            torch.zeros_like(per_level_topk_idxs))
        is_in_candidate = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(
            dim=-2)
        is_in_candidate = torch.where(is_in_candidate > 1, torch.zeros_like
            (is_in_candidate), is_in_candidate)
        is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
        start_idx = end_idx
    is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
    candidate_idxs = torch.cat(candidate_idxs, dim=-1)
    return is_in_candidate_list, candidate_idxs
