def get_metric(causal_score_list, ground_truth_list, config_dict,
    packed_data, split_idx_list=None):
    causal_label_list = []
    for group_idx, data in zip(range(0, len(causal_score_list), 4), packed_data
        ):
        group_score = causal_score_list[group_idx:group_idx + 4]
        k = len(data['cause_idx'])
        if k > 0:
            top_indices = np.argpartition(group_score, -k)[-k:]
            cur_pred_label = [int(i in top_indices) for i in range(4)]
        else:
            cur_pred_label = [0] * 4
        causal_label_list.extend(cur_pred_label)
    performance = get_scores(causal_label_list, ground_truth_list,
        split_idx_list)
    res = Result(config_dict, performance, causal_label_list)
    return res
