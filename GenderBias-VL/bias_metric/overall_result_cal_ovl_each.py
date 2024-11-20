def cal_ovl_each(self, model_name, bias_type):
    target_file = os.path.join(base_results_dir, model_name,
        f'{model_name}_{bias_type}.csv')
    data = read_file(target_file)
    data = sorted(data, key=lambda x: x['bias'], reverse=True)
    acc_mean = np.mean([row['acc'] for row in data])
    bias_mean = np.mean([abs(row['bias']) for row in data])
    bias_max = np.max([row['bias'] for row in data])
    ipst_mean = np.mean([row['ipss'] for row in data])
    acc_delta_mean = np.mean([row['acc_delta'] for row in data])
    result = OrderedDict({'ipss_ovl': ipst_mean, 'bias_ovl': bias_mean,
        'bias_max': bias_max, 'acc_ovl': acc_mean, 'acc_delta_ovl':
        acc_delta_mean})
    return result
