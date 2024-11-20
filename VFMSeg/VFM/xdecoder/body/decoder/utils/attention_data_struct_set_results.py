def set_results(self, results):
    for name in self.cross_attn_name:
        self.attn_variables[name].attn_mask = results['attn_mask'][:, self.
            query_index[name][0]:self.query_index[name][1]]
    for key in self.output:
        self.output[key].append(results[key])
