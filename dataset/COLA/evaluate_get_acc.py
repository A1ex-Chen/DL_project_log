def get_acc(causal_label_list, ground_truth_list):
    shared_positive_list = causal_label_list == ground_truth_list
    shared_count = np.sum(shared_positive_list)
    return shared_count / len(shared_positive_list)
