def set_low_quality_matches_(self, match_labels, matched_vals, matches, gt_ious
    ):
    for i in range(len(gt_ious)):
        match_labels[(matched_vals == gt_ious[i]) & (matches == i)] = 1
