def update_variables(self, output, mode):
    name_set = (self.self_attn_name if mode == 'self_attn' else self.
        cross_attn_name)
    for key in name_set:
        self.attn_variables[key].output = output[self.query_index[key][0]:
            self.query_index[key][1]]
