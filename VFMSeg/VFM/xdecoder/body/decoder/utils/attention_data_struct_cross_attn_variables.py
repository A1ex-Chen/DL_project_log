def cross_attn_variables(self):
    cross_attn_name = [key for key, value in self.cross_attn_dict.items() if
        value == True and key in self.attn_variables and (key not in self.
        flags or key in self.flags and self.flags[key] == True)]
    self.cross_attn_name = cross_attn_name
    output = torch.cat([self.attn_variables[name].output for name in
        cross_attn_name])
    pos_emb = torch.cat([self.attn_variables[name].pos for name in
        cross_attn_name])
    index = 0
    for name in cross_attn_name:
        self.query_index[name] = [index, index + self.attn_variables[name].
            output.shape[0]]
        index += self.attn_variables[name].output.shape[0]
    return output, pos_emb
