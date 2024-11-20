def get_latex_from_dict(metric_dict, split_name='all', contain_null=True):
    if split_name == 'all':
        ordered_name_list = ['valid_acc', 'valid_f1', 'valid_ma-f1', 'null',
            'test_acc', 'test_f1', 'test_ma-f1', 'null']
    elif split_name == 'valid':
        ordered_name_list = ['valid_acc', 'valid_f1', 'valid_ma-f1', 'null']
    elif split_name == 'test':
        ordered_name_list = ['test_acc', 'test_f1', 'test_ma-f1', 'null']
    else:
        raise ValueError('Wrong split name')
    if not contain_null:
        ordered_name_list = [metric for metric in ordered_name_list if 
            metric != 'null']
    latex_format_str = ''
    for cur_metric in ordered_name_list:
        if cur_metric != 'null':
            score = metric_dict[cur_metric]
            score *= 100
            score = round(score, 2)
            str_score = '{:.2f}'.format(score)
        else:
            str_score = '-'
        latex_format_str += '&' + str_score
    return latex_format_str
