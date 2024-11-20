def get_max_diff(self, ref_param, tst_param):
    max_abs_diff = max_rel_diff = 0
    for p_ref, p_tst in zip(ref_param, tst_param):
        max_abs_diff_p = (p_ref - p_tst).abs().max().item()
        max_rel_diff_p = ((p_ref - p_tst) / p_ref).abs().max().item()
        if max_abs_diff_p > max_abs_diff:
            max_abs_diff = max_abs_diff_p
        if max_rel_diff_p > max_rel_diff:
            max_rel_diff = max_rel_diff_p
    return max_abs_diff, max_rel_diff
