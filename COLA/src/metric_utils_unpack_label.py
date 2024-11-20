def unpack_label(packed_data, label_name='cause_idx'):
    ground_truth_list = []
    for idx, data_dict in enumerate(packed_data):
        pos_set = set(data_dict[label_name])
        cur_label = [int(i in pos_set) for i in range(4)]
        assert sum(cur_label) == len(pos_set), str(idx) + str(cur_label) + str(
            pos_set)
        ground_truth_list.extend(cur_label)
    return ground_truth_list
