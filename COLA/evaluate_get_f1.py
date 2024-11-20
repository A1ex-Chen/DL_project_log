def get_f1(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list * ground_truth_list
    shared_count = np.sum(shared_positive_list)
    p_count = np.sum(causal_label_list)
    r_count = np.sum(ground_truth_list)
    precision = shared_count / p_count if p_count != 0 else 0
    recall = shared_count / r_count if r_count != 0 else 0
    f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1
