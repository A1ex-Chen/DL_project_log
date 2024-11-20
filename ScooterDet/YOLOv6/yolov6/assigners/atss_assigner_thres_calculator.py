def thres_calculator(self, is_in_candidate, candidate_idxs, overlaps):
    n_bs_max_boxes = self.bs * self.n_max_boxes
    _candidate_overlaps = torch.where(is_in_candidate > 0, overlaps, torch.
        zeros_like(overlaps))
    candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
    assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=
        candidate_idxs.device)
    assist_idxs = assist_idxs[:, None]
    faltten_idxs = candidate_idxs + assist_idxs
    candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
    candidate_overlaps = candidate_overlaps.reshape([self.bs, self.
        n_max_boxes, -1])
    overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
    overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
    overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
    return overlaps_thr_per_gt, _candidate_overlaps
