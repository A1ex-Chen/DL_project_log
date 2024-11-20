def cal_prob_gap(self, base_value, cf_value):
    prob_gap_list = []
    prob_gap_list_id = []
    mean_prob_gap = 0
    for base, cf in zip(base_value, cf_value):
        gt_choice = base['gt_choice']
        prob_gap_list.append(base['probs'][gt_choice] - cf['probs'][gt_choice])
        prob_gap_list_id.append(OrderedDict({'id': base['id'], 'prob_gap': 
            base['probs'][gt_choice] - cf['probs'][gt_choice]}))
    mean_prob_gap = sum(prob_gap_list) / len(prob_gap_list) if len(
        prob_gap_list) != 0 else 0
    return mean_prob_gap, prob_gap_list_id
