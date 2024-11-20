def disentangled_att_bias(self, query_layer, key_layer, relative_pos,
    rel_embeddings, scale_factor):
    if relative_pos is None:
        q = query_layer.size(-2)
        relative_pos = build_relative_position(q, key_layer.size(-2),
            query_layer.device)
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    elif relative_pos.dim() != 4:
        raise ValueError(
            f'Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}'
            )
    att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.
        max_relative_positions)
    relative_pos = relative_pos.long().to(query_layer.device)
    rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:
        self.max_relative_positions + att_span, :].unsqueeze(0)
    if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
        pos_key_layer = self.pos_proj(rel_embeddings)
        pos_key_layer = self.transpose_for_scores(pos_key_layer)
    if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
        pos_query_layer = self.pos_q_proj(rel_embeddings)
        pos_query_layer = self.transpose_for_scores(pos_query_layer)
    score = 0
    if 'c2p' in self.pos_att_type:
        c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(
            c2p_pos, query_layer, relative_pos))
        score += c2p_att
    if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
        pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
        if query_layer.size(-2) != key_layer.size(-2):
            r_pos = build_relative_position(key_layer.size(-2), key_layer.
                size(-2), query_layer.device)
        else:
            r_pos = relative_pos
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        if query_layer.size(-2) != key_layer.size(-2):
            pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
    if 'p2c' in self.pos_att_type:
        p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_dynamic_expand(
            p2c_pos, query_layer, key_layer)).transpose(-1, -2)
        if query_layer.size(-2) != key_layer.size(-2):
            p2c_att = torch.gather(p2c_att, dim=-2, index=
                pos_dynamic_expand(pos_index, p2c_att, key_layer))
        score += p2c_att
    return score
