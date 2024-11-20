def calculate_metrics(self, base, cf):
    tf, ft, tt, ff = 0, 0, 0, 0
    tf_list, ft_list, tt_list, ff_list = [], [], [], []
    for p, l in zip(base, cf):
        assert str(p['id']) == str(l['id']), 'id should be the same'
        if p['metric_result'] is True:
            if l['metric_result'] is True:
                tt += 1
                tt_list.append(p['id'])
            else:
                tf += 1
                tf_list.append(p['id'])
        elif l['metric_result'] is True:
            ft += 1
            ft_list.append(p['id'])
        else:
            ff += 1
            ff_list.append(p['id'])
    return tf, ft, tt, ff, tf_list, ft_list, tt_list, ff_list
