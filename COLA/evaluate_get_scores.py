def get_scores(causal_label_list, ground_truth_list, split_idx_list=None):
    causal_label_list = np.array(causal_label_list)
    ground_truth_list = np.array(ground_truth_list)
    if split_idx_list is None:
        res = {}
        res['acc'] = get_acc(causal_label_list, ground_truth_list)
        res['ma-p'], res['ma-r'], res['ma-f1'], res['f1'] = get_macro_f1(
            causal_label_list, ground_truth_list)
    else:
        causal_label_list, ground_truth_list = pack_list(causal_label_list
            ), pack_list(ground_truth_list)
        valid_idx_list, test_idx_list = split_idx_list['valid'
            ], split_idx_list['test']
        eval_label_list = causal_label_list[valid_idx_list], causal_label_list[
            test_idx_list]
        eval_truth_list = ground_truth_list[valid_idx_list], ground_truth_list[
            test_idx_list]
        eval_label_list = [split_label_list.reshape(-1) for
            split_label_list in eval_label_list]
        eval_truth_list = [split_truth_list.reshape(-1) for
            split_truth_list in eval_truth_list]
        res = {}
        for name, label_list, truth_list in zip(['valid', 'test'],
            eval_label_list, eval_truth_list):
            res[f'{name}_acc'] = get_acc(label_list, truth_list)
            res[f'{name}_ma-p'], res[f'{name}_ma-r'], res[f'{name}_ma-f1'
                ], res[f'{name}_f1'] = get_macro_f1(label_list, truth_list)
    return res
