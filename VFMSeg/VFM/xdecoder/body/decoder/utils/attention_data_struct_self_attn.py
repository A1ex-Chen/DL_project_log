def self_attn(self, bs, num_heads):
    self_attn_name = [key for key, value in self.self_attn_dict.items() if 
        len(value) > 0 and key in self.attn_variables and (key not in self.
        flags or key in self.flags and self.flags[key] == True)]
    self.self_attn_name = self_attn_name
    output = torch.cat([self.attn_variables[name].output for name in
        self_attn_name])
    pos_emb = torch.cat([self.attn_variables[name].pos for name in
        self_attn_name])
    index = 0
    for name in self_attn_name:
        self.query_index[name] = [index, index + self.attn_variables[name].
            output.shape[0]]
        index += self.attn_variables[name].output.shape[0]
    self_attn_mask = torch.ones((bs, output.shape[0], output.shape[0]),
        dtype=torch.bool, device=output.device)
    self_attn_pair = []
    for key1, value in self.self_attn_dict.items():
        for key2 in value:
            if key1 not in self_attn_name or key2 not in self_attn_name:
                continue
            if (key1 in self.masking or key2 in self.masking) and key1 != key2:
                self_attn_pair += [[key1, key2]]
            self_attn_mask[:, self.query_index[key1][0]:self.query_index[
                key1][1], self.query_index[key2][0]:self.query_index[key2][1]
                ] = False
    for key in self.masking:
        if key in self_attn_name:
            self_attn_mask[:, self.query_index[key][0]:self.query_index[key
                ][1], self.query_index[key][0]:self.query_index[key][1]][self
                .attn_variables[key].masking] = True
            self_attn_mask[:, self.query_index[key][0]:self.query_index[key
                ][1], self.query_index[key][0]:self.query_index[key][1]
                ].transpose(1, 2)[self.attn_variables[key].masking] = True
    for key1, key2 in self_attn_pair:
        if key1 not in self_attn_name or key2 not in self_attn_name:
            continue
        if key1 in self.masking:
            self_attn_mask[:, self.query_index[key1][0]:self.query_index[
                key1][1], self.query_index[key2][0]:self.query_index[key2][1]][
                self.attn_variables[key1].masking] = True
        if key2 in self.masking:
            self_attn_mask[:, self.query_index[key1][0]:self.query_index[
                key1][1], self.query_index[key2][0]:self.query_index[key2][1]
                ].transpose(1, 2)[self.attn_variables[key2].masking] = True
    self_attn_mask = self_attn_mask.repeat_interleave(num_heads, dim=0)
    return output, pos_emb, self_attn_mask
