def write(self, result_all, file_name):
    model_data_list = []
    for model_name in self.model_names:
        merge_row = OrderedDict({'model_name': model_name})
        for bias_type in self.bias_types:
            temp = result_all[f'{model_name}_{bias_type}']
            for key, value in temp.items():
                merge_row[f'{bias_type}_{key}'] = value
        model_data_list.append(merge_row)
    self.write_csv(file_name, model_data_list)
