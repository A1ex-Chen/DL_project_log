def cal_ovl(self):
    result_all = OrderedDict()
    for model_name in self.model_names:
        for bias_type in self.bias_types:
            temp = self.cal_ovl_each(model_name, bias_type)
            result_all[f'{model_name}_{bias_type}'] = temp
    filename = os.path.join(exp_dir, 'overall', f'bias_overall.csv')
    self.write(result_all, filename)
