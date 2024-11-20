def get_macro_f1(causal_label_list, ground_truth_list):
    p1, r1, f1 = get_f1(causal_label_list, ground_truth_list)
    reversed_causal_label_list = 1 - causal_label_list
    reversed_ground_truth_list = 1 - ground_truth_list
    p0, r0, f0 = get_f1(reversed_causal_label_list, reversed_ground_truth_list)
    p = (p0 + p1) / 2
    r = (r0 + r1) / 2
    f = (f0 + f1) / 2
    return p, r, f, f1
