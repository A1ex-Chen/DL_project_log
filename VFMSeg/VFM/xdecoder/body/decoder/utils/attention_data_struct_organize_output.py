def organize_output(self):
    outputs = {}
    outputs['aux_outputs'] = [{} for i in range(self.num_layers)]
    for key, values in self.output.items():
        for _key, idx_name in zip(predict_name_matcher[key],
            predict_index_matcher[key]):
            if idx_name not in self.query_index:
                continue
            outputs[_key] = self.output[key][-1][:, self.query_index[
                idx_name][0]:self.query_index[idx_name][1]]
            for idx, aux_values in enumerate(self.output[key][:-1]):
                outputs['aux_outputs'][idx][_key] = aux_values[:, self.
                    query_index[idx_name][0]:self.query_index[idx_name][1]]
    return outputs
