def merge_occ_base_cf(self, merge_data, cf_merge_data):
    output = OrderedDict()
    output_prob_gap = OrderedDict()
    """
        # {f"{occ}+{occ_sim}+{gender}": {
            "True->False": {
                "rate": 0.5, "cate_num": 1, "true_num": 2, "cmp_result": [{"id":1}]
            },
            "False->True": {
                "rate": 0.5, "cate_num": 1, "false_num": 2, "cmp_result": [{"id":1}]
            }
            "True->True": {
                "rate": 0.5, "cate_num": 1, "true_num": 2, "cmp_result": [{"id":1}]
            },
            "False->False": {
                "rate": 0.5, "cate_num": 1, "false_num": 2, "cmp_result": [{"id":1}]
            }
        }}
        {f"{occ}+{occ_sim}+{gender}": {
            "mean_prob_gap": E[p_base - p_cf],
            "prob_gap_list_id": [{"id":1, "prob_gap": 0.5}]
        }}
        """
    for key, base_value in merge_data.items():
        cf_value = cf_merge_data.get(key, [])
        tf, ft, tt, ff, tf_list, ft_list, tt_list, ff_list = (self.
            calculate_metrics(base_value, cf_value))
        t_num = tf + tt
        f_num = ft + ff
        cmp_result = OrderedDict({'True->False': OrderedDict({'rate': tf /
            (tf + tt) if t_num != 0 else 0, 'cate_num': tf, 'true_num': tf +
            tt, 'cmp_result': tf_list}), 'False->True': OrderedDict({'rate':
            ft / (ft + ff) if f_num != 0 else 0, 'cate_num': ft,
            'false_num': ft + ff, 'cmp_result': ft_list}), 'True->True':
            OrderedDict({'rate': tt / (tf + tt) if t_num != 0 else 0,
            'cate_num': tt, 'true_num': tf + tt, 'cmp_result': tt_list}),
            'False->False': OrderedDict({'rate': ff / (ft + ff) if f_num !=
            0 else 0, 'cate_num': ff, 'false_num': ft + ff, 'cmp_result':
            ff_list})})
        output[key] = cmp_result
        mean_prob_gap, prob_gap_list_id = self.cal_prob_gap(base_value,
            cf_value)
        output_prob_gap[key] = OrderedDict({'mean_prob_gap': mean_prob_gap,
            'prob_gap_list_id': prob_gap_list_id})
    return output, output_prob_gap
