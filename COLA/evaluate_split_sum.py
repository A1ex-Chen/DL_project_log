def split_sum(score_dict, split_name):
    split_name_list = ['valid', 'test'] if split_name == 'all' else [split_name
        ]
    score_sum = 0
    for cur_sn in split_name_list:
        score_sum += score_dict[f'{cur_sn}_f1']
        score_sum += score_dict[f'{cur_sn}_ma-f1']
        score_sum += score_dict[f'{cur_sn}_acc']
    return score_sum
