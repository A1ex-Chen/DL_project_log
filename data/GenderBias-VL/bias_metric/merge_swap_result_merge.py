def merge(result, result_swap, model_name, bias_type):
    merge_data = []
    result_data = read_file(result)
    result_swap_data = read_file(result_swap)
    for i in range(0, len(result_data)):
        result_row = result_data[i]
        result_swap_row = result_swap_data[i]
        merge_row = copy.deepcopy(result_row)
        for key, value in result_row.items():
            if isinstance(value, (int, float)):
                merge_row[key] = (value + result_swap_row[key]) / 2
        merge_row['acc_delta'] = abs(result_row['occtm_acc'] -
            result_swap_row['occtm_acc']) + abs(result_row['occtf_acc'] -
            result_swap_row['occtf_acc'])
        merge_row['acc_delta'] = merge_row['acc_delta'] / 2
        merge_data.append(merge_row)
    merge_data = sorted(merge_data, key=lambda x: x['bias'], reverse=True)
    occtm_list, occtf_list = cal_micro_occ_bias(merge_data)
    filednames = 'occ,micro_bias'.split(',')
    filename = f'{model_name}_{bias_type}_occtm.csv'
    filename = os.path.join(exp_dir, 'merge_micro_occ', model_name, filename)
    write_csv(filename, occtm_list, mode='w', fieldnames=filednames)
    filename = f'{model_name}_{bias_type}_occtf.csv'
    filename = os.path.join(exp_dir, 'merge_micro_occ', model_name, filename)
    write_csv(filename, occtf_list, mode='w', fieldnames=filednames)
    mean_acc_delta = np.mean([row['acc_delta'] for row in merge_data])
    print(f'{model_name} {bias_type} mean acc delta: {mean_acc_delta:.2f}')
    filename = f'{model_name}_{bias_type}.csv'
    filename = os.path.join(exp_dir, 'merge_bias_probability', model_name,
        filename)
    write_csv(filename, merge_data, mode='w')
